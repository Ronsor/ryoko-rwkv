# Copyright (C) Ronsor Labs and the Ryoko AI Production Committee.
# Copyright (C) BlinkDL.
#
# This software may be used and distributed according to the terms of the
# Apache License version 2.0, a copy of which may be found in the LICENSE file
# at the root of this repository.

# RNN variant of the model. Ported from RWKV-LM.

import gc, math, os, torch
import torch.nn as nn
from torch.nn import functional as F
from types import SimpleNamespace
from .model import RyokoConfig, RMSNorm

RWKV_RESCALE_LAYER = 6

class RyokoRNN(nn.Module):
  def __init__(self, config, weights, dtype=torch.float32, device="cpu", debug=False):
    super().__init__()

    self.config = config
    self.dtype = dtype
    self.device = torch.device(device)

    with torch.no_grad():
      w = weights

      print_need_newline = False
      for x in w:
        block_id = 0
        if 'blocks.' in x:
          block_id = int(x.split('.')[1])
        elif 'att.output.weight' in x or 'ffn.value.weight' in x:
          w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))

        if '.time_' in x:
          w[x] = w[x].squeeze()
        if '.time_decay' in x:
          w[x] = w[x].float()
          w[x] = -torch.exp(w[x])
        elif '.time_first' in x:
          w[x] = w[x].float()
        else:
          if self.dtype != None:
            w[x] = w[x].to(dtype=dtype)

        w[x].requires_grad = False
        if device is not None and x != 'emb.weight':
          w[x] = w[x].to(device=device)

        if (('blocks.' not in x) or ('blocks.0.' in x)) and debug:
          if print_need_newline:
            print('\n', end = '')
            print_need_newline = False
            print(x.ljust(40), str(w[x].dtype).replace('torch.', '').ljust(10), w[x].device)
          else:
            print_need_newline = True
            print('.', end = '', flush = True)

    self.w = SimpleNamespace()
    for x in w:
      xx = x.split('.')
      here = self.w
      for i in range(len(xx)):
        if xx[i].isdigit():
          ii = int(xx[i])
          if ii not in here:
            here[ii] = SimpleNamespace()
          here = here[ii]
        else:
          if i == len(xx) - 1:
            setattr(here, xx[i], w[x])
          elif not hasattr(here, xx[i]):
            if xx[i+1].isdigit():
              setattr(here, xx[i], {})
            else:
              setattr(here, xx[i], SimpleNamespace())
          here = getattr(here, xx[i])

    self.eval()

  def LN(self, x, w):
    eps = 1e-5
    norm_x = torch.mean(x * x, dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(norm_x + eps)
    return w.weight * x_normed

  def FF(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
    if self.dtype != torch.float32:
      xk = x * time_mix_k + state[5*i+0].to(dtype=self.dtype) * (1 - time_mix_k)
      xr = x * time_mix_r + state[5*i+0].to(dtype=self.dtype) * (1 - time_mix_r)
      state[5*i+0] = x.float()
    else:
      xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
      xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
      state[5*i+0] = x

    r = torch.sigmoid(rw @ xr)
    k = torch.square(torch.relu(kw @ xk))
    kv = vw @ k

    return r * kv

  def SA(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
    if self.dtype != torch.float32:
      xk = x * time_mix_k + state[5*i+1].to(dtype=self.dtype) * (1 - time_mix_k)
      xv = x * time_mix_v + state[5*i+1].to(dtype=self.dtype) * (1 - time_mix_v)
      xr = x * time_mix_r + state[5*i+1].to(dtype=self.dtype) * (1 - time_mix_r)
      state[5*i+1] = x.float()
    else:
      xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
      xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
      xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
      state[5*i+1] = x

    r = torch.sigmoid(rw @ xr)
    k = kw @ xk
    v = vw @ xv

    if self.dtype != torch.float32:
      kk = k.float()
      vv = v.float()
    else:
      kk = k
      vv = v
    aa = state[5*i+2]
    bb = state[5*i+3]
    pp = state[5*i+4]
    ww = time_first + kk
    p = torch.maximum(pp, ww)
    e1 = torch.exp(pp - p)
    e2 = torch.exp(ww - p)
    a = e1 * aa + e2 * vv
    b = e1 * bb + e2
    ww = pp + time_decay
    p = torch.maximum(ww, kk)
    e1 = torch.exp(ww - p)
    e2 = torch.exp(kk - p)
    state[5*i+2] = e1 * aa + e2 * vv
    state[5*i+3] = e1 * bb + e2
    state[5*i+4] = p
    if self.dtype != torch.float32:
      wkv = (a / b).to(dtype=self.dtype)
    else:
      wkv = a / b

    return ow @ (r * wkv)

  def forward(self, ctx, state, preprocess_only: bool = False):
    with torch.no_grad():
      w = self.w
      config = self.config

      x = w.emb.weight[ctx[-1]]
      if self.device != x.device:
        x = x.to(device=self.device)

      if state == None:
        state = torch.zeros(config.n_layer * 5, config.n_embd, device=self.device)
        for i in range(config.n_layer):
          state[5*i+4] -= 1e30

      for i in range(config.n_layer):
        if i == 0:
          x = self.LN(x, w.blocks[i].ln0)

        ww = w.blocks[i].att
        x = x + self.SA(self.LN(x, w.blocks[i].ln1), state, i,
          ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay,
          ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)

        ww = w.blocks[i].ffn
        x = x + self.FF(self.LN(x, w.blocks[i].ln2), state, i,
          ww.time_mix_k, ww.time_mix_r,
          ww.key.weight, ww.value.weight, ww.receptance.weight)

        if (i+1) % RWKV_RESCALE_LAYER == 0:
          x = x / 2

      if preprocess_only:
        return state

      x = self.LN(x, w.ln_out)
      x = w.head.weight @ x

      return x.float(), state

def FastRyokoRNN(*v, **kv):
  return torch.jit.script(RyokoRNN(*v, **kv))
