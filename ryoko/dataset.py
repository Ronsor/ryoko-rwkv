# Copyright (C) Ronsor Labs and the Ryoko AI Production Committee.
#
# This software may be used and distributed according to the terms of the
# Apache License version 2.0, a copy of which may be found in the LICENSE file
# at the root of this repository.

import torch
import numpy as np
import torch.utils.data as TUD
from dataclasses import dataclass

class RandomDataset(TUD.Dataset):
  def __init__(self, data, model_config, train_config):
    self.data = data

    self.model_config = model_config
    self.train_config = train_config

  def __len__(self):
    return self.train_config.epoch_steps * 16

  def __getitem__(self, idx):
    ctx_len = self.model_config.ctx_len

    i = np.random.randint(0, len(self.data) - 1)
    dix = self.data[i:i + ctx_len + 1]

    x = torch.tensor(dix[:-1], dtype=torch.long)
    y = torch.tensor(dix[1:], dtype=torch.long)
    return x, y

class SequentialDataset(TUD.Dataset):
  def __init__(self, data, model_config, train_config, offset: int = 0):
    self.data = data

    self.model_config = model_config
    self.train_config = train_config
    self.offset = offset

  def __len__(self):
    return len(data) - 1

  def __getitem__(self, idx):
    ctx_len = self.model_config.ctx_len

    i = self.offset
    self.offset += 1
    dix = self.data[i:i + ctx_len + 1]

    x = torch.tensor(dix[:-1], dtype=torch.long)
    y = torch.tensor(dix[1:], dtype=torch.long)
    return x, y
