# Copyright (C) Ronsor Labs and the Ryoko AI Production Committee.
#
# This software may be used and distributed according to the terms of the
# Apache License version 2.0, a copy of which may be found in the LICENSE file
# at the root of this repository.

import json, os, os.path, time, torch
from datetime import datetime, timezone
import lightning as L
from dataclasses import dataclass
from math import exp, log

@dataclass
class TrainConfig:
  lr_init: float    = None
  lr_final: float   = None
  warmup_steps: int = 0

  epoch_begin: int = 0
  epoch_steps: int = 0
  epoch_count: int = 1

  betas: any      = (0.9, 0.99)
  adam_eps: float = 1e-8

  grad_cp: bool   = True

  project_init: bool     = False
  project_path: str      = None
  project_ckpt_file: str = 'ryoko-{epoch}.pth'

@dataclass
class TrainLossStats:
  current: float = None
  epoch: int     = 0
  count: float   = 0.0
  sum: float     = 0.0

class TrainCallback(L.Callback):
  def __init__(self, train_config, model_config=None, extra_log=None, extra_log_init=None):
    super().__init__()
    self.train_config = train_config
    self.model_config = model_config
    self.extra_log = extra_log
    self.extra_log_init = extra_log_init
    self._saved = False

  def on_train_batch_start(self, trainer, module, batch, batch_idx):
    tcfg = self.train_config

    step = trainer.global_step + tcfg.epoch_begin * tcfg.epoch_steps
    w_step = tcfg.warmup_steps

    if tcfg.lr_init == tcfg.lr_final or tcfg.epoch_count == 0:
      # Constant learning rate
      lr = tcfg.lr_init
      if trainer.global_step < w_step:
        lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)
    else:
      # Variable learning rate
      decay_total = tcfg.epoch_count * tcfg.epoch_steps
      progress = (step - w_step + 1) / (decay_total - w_step)
      progress = min(1, max(0, progress))

      if tcfg.lr_init == 0 or tcfg.lr_final == 0:
        # Linear decay
        lr = tcfg.lr_init + (tcfg.lr_final - tcfg.lr_init) * progress
      else:
        # Exponential decay
        lr = tcfg.lr_init * exp(log(tcfg.lr_final / tcfg.lr_init) * pow(progress, 1))

    # Update learning rate
    for param_group in trainer.optimizers[0].param_groups:
       param_group["lr"] = lr

    trainer.current_lr = lr
    if trainer.loss_stats is None:
      trainer.loss_stats = TrainLossStats()

    if self._saved or not trainer.is_global_zero: return
    self._save_log(trainer, module, True)
    self._saved = True

  def on_train_batch_end(self, trainer, module, outputst, batch, batch_idx):
    if not trainer.is_global_zero: return
    if not hasattr(trainer, 'model_loss_all'): return

    tcfg = self.train_config
    trainer.loss_stats.current = trainer.model_loss_all['loss'].float().mean().item()
    trainer.loss_stats.sum += trainer.loss_stats.current
    trainer.loss_stats.count += 1
    trainer.loss_stats.epoch = trainer.loss_stats.sum / trainer.loss_stats.count

    self.log("lr", trainer.current_lr, prog_bar=True, on_step=True)
    self.log("loss", trainer.loss_stats.epoch, prog_bar=True, on_step=True)

    try:
      cost = (time.time_ns() - self._clock) / 1e9
      kt_s = (self.model_config.ctx_len * 12) / cost / 1000
      self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
    except:
      pass
    self._clock = time.time_ns()

  def on_train_epoch_end(self, trainer, module):
    if not trainer.is_global_zero: return
    self._save_log(trainer, module)

  def _save_log(self, trainer, module, initial=False):
    tcfg = self.train_config

    if tcfg.project_path is None: return
    os.makedirs(tcfg.project_path, exist_ok=True)

    now = datetime.now(tz=timezone.utc)
    now_ts = now.timestamp()
    now_str = now.strftime('%FT%T%z')

    with open(os.path.join(tcfg.project_path, 'train_log.txt'), 'a') as fp:
      entry = {
        '@': now_str,
        'ts': now_ts,
        'initial': initial,
        'epoch': tcfg.epoch_begin + trainer.current_epoch,
        'lr': trainer.current_lr,
        'loss': trainer.loss_stats.__dict__,
      }
      if self.extra_log is not None:
        entry['extra_log'] = self.extra_log
      if initial:
        entry['train_config'] = self.train_config.__dict__ if self.train_config else None
        entry['model_config'] = self.model_config.__dict__ if self.model_config else None
        entry['extra_log_init'] = self.extra_log_init
      fp.write(f'{json.dumps(entry, ensure_ascii=False)}\n')

      del entry['@']
      del entry['ts']
      print(now_str, entry)

    if initial and tcfg.epoch_begin != 0: return

    ckpt_file_vars = {
      'epoch': tcfg.epoch_begin + trainer.current_epoch if not initial else 'init',
      'train': self.train_config,
      'model': self.model_config,
      'trainer': trainer,
    }
    ckpt_file_name = tcfg.project_ckpt_file.format(**ckpt_file_vars)

    # TODO: safetensors
    torch.save(module.state_dict(), os.path.join(tcfg.project_path, ckpt_file_name))
