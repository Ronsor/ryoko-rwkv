# Copyright (C) Ronsor Labs and the Ryoko AI Production Committee.
# Copyright (C) BlinkDL.
#
# This software may be used and distributed according to the terms of the
# Apache License version 2.0, a copy of which may be found in the LICENSE file
# at the root of this repository.
#
# Heavily derived from the RWKV language model at
# <https://github.com/BlinkDL/RWKV-LM>.

# Major changes:
# - Use LLaMA-style RMSNorm instead of LayerNorm
# - TODO: the rest

import math, os, torch
import lightning as L
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from .util import any_in

try:
  import deepspeed
  from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
  from lightning.pytorch.strategies import DeepSpeedStrategy
except:
  pass

@dataclass
class RyokoConfig:
  ctx_len: int    = 1024
  n_layer: int    = 12
  n_embd: int     = 768

  vocab_size:        int = 30000
  output_vocab_size: int = None

  # These options are considered even more experimental than the rest.
  attention: bool = False
  pos_emb: bool   = False
  dropout: float  = None

  @property
  def dim_att(self): return self.n_embd
  @property
  def dim_ffn(self): return self.n_embd * 4

class RMSNorm(nn.Module):
  """LLaMA-style Root Mean Square Layer Normalization

  It's faster than LayerNorm (ostensibly). Don't think I've actually tested
  that.
  """

  def __init__(self, size: int, dim: int = -1, eps: float = 1e-5):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(size))
    self.dim = dim
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
    x_normed = x * torch.rsqrt(norm_x + self.eps)
    return self.weight * x_normed

class RWKV_TimeMix(nn.Module):
  def __init__(self, config, layer_id):
    super().__init__()
    self.config = config
    self.layer_id = layer_id
    self.ctx_len = config.ctx_len
    self.n_embd = config.n_embd

    with torch.no_grad():  # fancy init
      ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
      ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
      ddd = torch.ones(1, 1, config.n_embd)
      for i in range(config.n_embd):
        ddd[0, 0, i] = i / config.n_embd

      # fancy time_decay
      decay_speed = torch.ones(config.dim_att)
      for h in range(config.dim_att):
        decay_speed[h] = -5 + 8 * (h / (config.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
      self.time_decay = nn.Parameter(decay_speed)

      # fancy time_first
      zigzag = torch.tensor([(i + 1) % 3 - 1
                             for i in range(config.dim_att)]) * 0.5
      self.time_first = nn.Parameter(torch.ones(config.dim_att) *
                                     math.log(0.3) + zigzag)

      # fancy time_mix
      self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
      self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) +
                                     0.3 * ratio_0_to_1)
      self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

    self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
    self.key = nn.Linear(config.n_embd, config.dim_att, bias=False)
    self.value = nn.Linear(config.n_embd, config.dim_att, bias=False)
    self.receptance = nn.Linear(config.n_embd, config.dim_att, bias=False)
    self.output = nn.Linear(config.dim_att, config.n_embd, bias=False)

    if config.attention:
      self.register_buffer("att_mask", torch.tril(
                           torch.ones(config.ctx_len, config.ctx_len)))
      d_qkv = config.n_embd // 16
      self.qq = nn.Linear(config.n_embd, d_qkv, bias=False)
      self.kk = nn.Linear(config.n_embd, d_qkv, bias=False)
      self.vv = nn.Linear(config.n_embd, d_qkv, bias=False)
      self.oo = nn.Linear(d_qkv, config.n_embd, bias=False)
      with torch.no_grad():
        self.time_mix_qq = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        self.time_mix_kk = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        self.time_mix_vv = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) +
                                        0.3 * ratio_0_to_1)
      self.use_forward = self.forward_att
    else:
      self.use_forward = self.forward_no_att

  # TODO: optimize more
  def WKV(self, k, v):
    decay = -torch.exp(self.time_decay)
    assert k.shape == v.shape
    assert 2 <= len(k.shape) <= 3
    B = (k.shape[-3],) if len(k.shape) > 2 else ()
    T = k.shape[-2]
    D = k.shape[-1]
    curve = torch.maximum(torch.zeros(1,device=k.device),
                         (torch.arange(T,device=k.device) - 1))[:,None] * decay
    curve[0] += self.time_first
    dx = curve.exp()
    kx = k.exp()
    zeros = torch.zeros(B+(D, T-1), dtype=kx.dtype, device=kx.device)
    wr = torch.conv1d(
         torch.cat([zeros, kx.transpose(-1,-2)],-1),
         dx.flip(0).transpose(0,1)[:,None,:], groups=D).transpose(-1,-2)
    vr = torch.conv1d(
         torch.cat([zeros, (kx * v).transpose(-1,-2)],-1),
         dx.flip(0).transpose(0,1)[:,None,:], groups=D).transpose(-1,-2)
    return vr / wr

  def QKV(self, q, k, v):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.att_mask == 0, -math.inf)
    att = F.softmax(att, dim = -1)
    x = att @ v
    return x

  def jit_func(self, x):
    xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
    xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
    xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
    xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
    k = self.key(xk)
    v = self.value(xv)
    r = self.receptance(xr)
    sr = torch.sigmoid(r)
    return sr, k, v

  def jit_funcQKV(self, x):
    xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
    xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
    xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
    xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
    xqq = x * self.time_mix_qq + xx * (1 - self.time_mix_qq)
    xkk = x * self.time_mix_kk + xx * (1 - self.time_mix_kk)
    xvv = x * self.time_mix_vv + xx * (1 - self.time_mix_vv)
    k = self.key(xk)
    v = self.value(xv)
    r = self.receptance(xr)
    sr = torch.sigmoid(r)
    qq = self.qq(xqq)
    kk = self.kk(xkk)
    vv = self.vv(xvv)
    return sr, k, v, qq, kk, vv

  def forward_no_att(self, x):
    B, T, C = x.size()  # x = (Batch,Time,Channel)
    sr, k, v = self.jit_func(x)

    rwkv = sr * self.WKV(k, v)
    return self.output(rwkv)

  def forward_att(self, x):
    B, T, C = x.size()  # x = (Batch,Time,Channel)
    sr, k, v, qq, kk, vv = self.jit_funcQKV(x)

    rwkv = sr * self.WKV(k, v)
    rwkv = self.output(rwkv) + self.oo(self.QKV(qq, kk, vv))
    return rwkv

  def forward(self, x):
    return self.use_forward(x)

class RWKV_ChannelMix(nn.Module):
  def __init__(self, config, layer_id):
    super().__init__()
    self.config = config
    self.layer_id = layer_id
    self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

    with torch.no_grad():  # fancy init of time_mix
      ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
      ddd = torch.ones(1, 1, config.n_embd)
      for i in range(config.n_embd):
        ddd[0, 0, i] = i / config.n_embd
      self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
      self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

    self.key = nn.Linear(config.n_embd, config.dim_ffn, bias=False)
    self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
    self.value = nn.Linear(config.dim_ffn, config.n_embd, bias=False)

  def forward(self, x):
    xx = self.time_shift(x)
    xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
    xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
    k = self.key(xk)
    k = torch.square(torch.relu(k))
    kv = self.value(k)
    return torch.sigmoid(self.receptance(xr)) * kv

class Block(nn.Module):
  def __init__(self, config, layer_id):
    super().__init__()
    self.config = config
    self.layer_id = layer_id

    self.ln1 = RMSNorm(config.n_embd)
    self.ln2 = RMSNorm(config.n_embd)

    if self.layer_id == 0:
      self.ln0 = RMSNorm(config.n_embd)

    self.att = RWKV_TimeMix(config, layer_id)
    self.ffn = RWKV_ChannelMix(config, layer_id)

  def forward(self, x, x_emb=None):
    config = self.config
    B, T, C = x.size()
    if self.layer_id == 0:
      x = self.ln0(x)

    x = x + self.att(self.ln1(x))
    x = x + self.ffn(self.ln2(x))

    return x

class L2Wrap(torch.autograd.Function):
  @staticmethod
  def forward(ctx, loss, y):
    ctx.save_for_backward(y)
    return loss

  @staticmethod
  def backward(ctx, grad_output):
    y = ctx.saved_tensors[0]
    factor = 1e-4 / (y.shape[0] * y.shape[1])
    maxx, ids = torch.max(y, -1, keepdim=True)

    gy = torch.zeros_like(y)
    gy.scatter_(-1, ids, maxx * factor)
    return grad_output, gy

class Ryoko(L.LightningModule):
  def __init__(self, config, train_config=None):
    super().__init__()
    self.config = config
    self.train_config = train_config

    self.emb = nn.Embedding(config.vocab_size, config.n_embd)

    if config.dropout is not None:
      self.drop = nn.Dropout(config.dropout)

    if config.pos_emb:
      self.pos_emb = nn.Embedding(config.ctx_len, config.n_embd)

    self.blocks = nn.ModuleList([
      Block(config, i) for i in range(config.n_layer)
    ])

    self.ln_out = RMSNorm(config.n_embd)

    head_vocab_size = (config.output_vocab_size
                       if config.output_vocab_size is not None
                       else config.vocab_size)
    self.head = nn.Linear(config.n_embd, head_vocab_size, bias=False)

  def forward(self, idx):
    config = self.config

    B, T = idx.size()
    assert T <= config.ctx_len, "Cannot forward: ctx_len exceeded"

    x = self.emb(idx)

    if config.pos_emb:
      pos = torch.arange(0, T, dtype=torch.long, device=self.idx.device()).unsqueeze(0)
      x = x + self.pos_emb(pos)

    if config.dropout is not None:
      x = self.drop(x)

    for block in self.blocks:
      if self.train_config is not None and self.train_config.grad_cp:
        x = deepspeed.checkpointing.checkpoint(block, x)
      else:
        x = block(x_emb)

    x = self.ln_out(x)
    x = self.head(x)

    return x

  def configure_optimizers(self):
    optim_groups = [
      {"params": [p for n, p in self.named_parameters()], "weight_decay": 0.0},
    ]

    tcfg = self.train_config
    assert tcfg is not None, "Cannot configure optimizers: missing training config"

    adam_config = dict(
      lr=tcfg.lr_init, betas=tcfg.betas, eps=tcfg.adam_eps,
      bias_correction=True, weight_decay=0, amsgrad=False,
    )

    offload = False
    strategy = self.trainer.strategy
    if isinstance(strategy, DeepSpeedStrategy):
      strat_cfg = strategy.config["zero_optimization"]
      offload = strat_cfg.get("offload_optimizer") or strat_cfg.get("offload_param")
    if offload:
      return DeepSpeedCPUAdam(optim_groups, adamw_mode=False, **adam_config)

    return FusedAdam(optim_groups, adam_w_mode=False, **adam_config)

  def training_step(self, batch, batch_idx):
    idx, targets = batch
    logits = self(idx)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    return L2Wrap.apply(loss, logits)

  def on_train_batch_end(self, outputs, batch, batch_idx):
    if self.trainer.is_global_zero:
      self.trainer.model_loss_all = self.all_gather(outputs)

  def init_weights(self, lr_init=None, device=None, dtype=None, debug=False):
    m = self.state_dict()
    param_count = 0

    if lr_init is None:
      lr_init = 1e-4 if self.train_config is None else self.train_config.lr_init

    for n in m:
      p = m[n]
      shape = p.shape

      if any_in(n, ["ln_", ".ln", "time_", "_mask", ".mask."]):
        continue

      gain = 1.0
      scale = 1.0
      if n == "emb.weight":
        scale = -1 * lr_init
      else:
        if shape[0] > shape[1]:
          gain = math.sqrt(shape[0] / shape[1])
        if any_in(n, [".att.key.", ".att.receptance.", ".att.output.", ".att.key.",
                      ".ffn.value.", ".ffn.receptance.", ".oo."]):
          scale = 0
        if n == "head.weight":
            scale = 0.5

      if debug:
        print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

      param_count += shape[0] * shape[1]

      if scale == 0:
        nn.init.zeros_(m[n])
      elif scale < 0:
        nn.init.uniform_(m[n], a=scale, b=-scale)
      else:
        nn.init.orthogonal_(m[n], gain=gain * scale)

      if device != None:
        m[n] = m[n].to(device=device)
      if dtype != None:
        m[n] = m[n].to(dtype=dtype)

    return param_count

  @property
  def device(self):
    return next(self.parameters()).device

def FastRyoko(*v, **kv):
  return torch.jit.script(Ryoko(*v, **kv))
