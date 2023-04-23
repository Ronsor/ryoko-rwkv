# Copyright (C) Ronsor Labs and the Ryoko AI Production Committee.
#
# This software may be used and distributed according to the terms of the
# Apache License version 2.0, a copy of which may be found in the LICENSE file
# at the root of this repository.

import os
from sentencepiece import SentencePieceProcessor
from typing import List, NamedTuple

class Token(NamedTuple):
  id: int
  piece: str
  is_control: bool = False

  def __str__(self): return self.piece

class Tokenizer:
  def __init__(self, model_file: str):
    assert os.path.isfile(model_file), f"No such model: {model_file}"

    self.sp_model = SentencePieceProcessor(model_file=model_file)

    self.vocab_size = self.sp_model.vocab_size()
    self.unk_id = self.sp_model.unk_id()
    self.bos_id = self.sp_model.bos_id()
    self.eos_id = self.sp_model.eos_id()
    self.pad_id = self.sp_model.pad_id()

  def get_token_info(self, t: any) -> Token:
    if type(t) is str:
      s = t
      t = self.sp_model.piece_to_id(s)
    else:
      s = self.sp_model.id_to_piece(s)

    if t == self.unk_id:
      s = ""

    return Token(
      id=t,
      piece=s,
      is_control=self.sp_model.is_control(t)
    )

  def encode(self, s, add_bos=True, add_eos=False) -> List[int]:
    return self.sp_model.encode(s.encode('utf-8'),
                                add_bos=add_bos, add_eos=add_eos)

  def decode(self, t: List[int]) -> str:
    return self.sp_model.decode(t)

