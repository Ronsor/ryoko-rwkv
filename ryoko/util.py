# Copyright (C) Ronsor Labs and the Ryoko AI Production Committee.
#
# This software may be used and distributed according to the terms of the
# Apache License version 2.0, a copy of which may be found in the LICENSE file
# at the root of this repository.

from functools import reduce
from types import SimpleNamespace

def any_in(s, l):
  for n in l:
    if n in s: return True
  return False
