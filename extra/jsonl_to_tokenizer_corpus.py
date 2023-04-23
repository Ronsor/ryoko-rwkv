#!/usr/bin/env python3
# Copyright (C) Ronsor Labs and the Ryoko AI Production Committee.
#
# This software may be used and distributed according to the terms of the
# Apache License version 2.0, a copy of which may be found in the LICENSE file
# at the root of this repository.

# Convert a JSONL dataset to a text corpus that can be used by SentencePiece.

import argparse, json, os, random, sys, struct, time

VERBOSE = False
def eprint(*v, **kv): print(*v, file=sys.stderr, **kv)
def vprint(*v, **kv):
  if VERBOSE:
    print(*v, file=sys.stderr, **kv)
def chunk(line, n): return [line[i:i+n] for i in range(0, len(line), n)]

class LineIndex:
  MAGIC = b'LNDX'
  STRUCT = '<QQ'

  def __init__(self, path=None):
    self.index = []
    if path != None:
      with open(path, 'rb') as fp:
        self.load(fp)

  def load(self, fp):
    sig = fp.read(4)
    if sig != LineIndex.MAGIC: raise Exception("Not a LineIndex file")
    while True:
      entry = fp.read(16)
      if len(entry) == 0: return
      if len(entry) != 16: raise Exception("Corrupt LineIndex file")
      off1, off2 = struct.unpack(LineIndex.STRUCT, entry)
      self.index.append((off1, off2))

  def dump(self, fp):
    fp.write(LineIndex.MAGIC)
    for off1, off2 in self.index:
      fp.write(struct.pack(LineIndex.STRUCT, off1, off2))

  def train(self, fp, tqdm=None):
    pbar = tqdm() if tqdm else None
    while True:
      off1 = fp.tell()
      if len(fp.readline()) == 0: break
      off2 = fp.tell()
      self.index.append((off1, off2))
      if pbar is not None:
        pbar.update(1)
    if pbar is not None: pbar.close()

  def lookup(self, lineno):
    return self.index[lineno]

  def total(self):
    return len(self.index)

  def readline(self, fp, lineno):
    off1, off2 = self.lookup(lineno)
    fp.seek(off1)
    return fp.read(off2 - off1)

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Be verbose')
parser.add_argument('-P', '--progress', default=False, action='store_true',
                    help='Show progress')
parser.add_argument('-o', '--output', default='tokenizer_corpus.txt',
                    help='Corpus output file')
parser.add_argument('-W', '--wrap_lines', default=4192,
                    help='Wrap corpus text lines after specified number of '
                         'characters')
parser.add_argument('-s', '--skip_malformed', default=False,
                    action='store_true',
                    help='Skip malformed JSON lines')
parser.add_argument('-S', '--shuffle_seed', default=42, type=int,
                    help='Shuffle lines deterministically. '
                         'Set to 0 to disable or -1 for a random seed')
parser.add_argument('-I', '--line_index', default=False, action='store_true',
                    help='Create line indexing file. An in-memory index will '
                         'be used regardless of this option if shuffling is '
                         'enabled')
parser.add_argument('--line_index_threads', default=1,
                    help='Use multiple threads for line indexing')
parser.add_argument('-F', '--json_fields', default=['text', 'content'],
                    type=(lambda x: x.split(',')),
                    help='List of JSON field(s) that contain corpus text')
parser.add_argument('input_files', nargs='+', help='Input JSON lines file')

args = parser.parse_args()
VERBOSE = args.verbose

if args.progress:
  try:
    from tqdm import tqdm
  except Exception as exc:
    eprint('You need to install tqdm to use the --progress option.')
    eprint('Exception details:', exc)
    sys.exit(1)
else:
  tqdm = lambda x: x

if args.line_index_threads > 1:
  try:
    from multiprocessing.pool import ThreadPool
    from p_tqdm import p_umap
  except Exception as exc:
    eprint('You need to install p_tqdm to set line_index_threads > 1.')
    eprint('Exception details:', exc)
    sys.exit(1)
  eprint('Warning:', f'line_index_threads > 1 is not yet supported')
  # TODO: support multiple threads

input_fps = []
for path in args.input_files:
  try:
    input_fps.append(open(path, 'rb'))
  except Exception as exc:
    eprint(f'Opening input file: {path}:', exc)
    sys.exit(1)

assert len(input_fps) == len(args.input_files)

line_indexes = None
if args.shuffle_seed != 0 or args.line_index:
  bench_start = time.monotonic()

  line_indexes = {}
  existing_fps = {}
  if args.line_index:
    idx_fps = []
    for path in args.input_files:
      try:
        if os.path.isfile(path + '.idx'):
          existing_fps[path] = open(path + '.idx', 'rb')
          idx_fps.append(True)
        else:
          idx_fps.append(open(path + '.idx', 'wb'))
      except Exception as exc:
        eprint(f'Opening LineIndex output file: {path}:', exc)
        sys.exit(1)
  else:
    idx_fps = [None] * len(args.input_files)

  for path, infp, idxfp in tqdm(zip(args.input_files, input_fps, idx_fps)):
    if args.progress: vprint()
    if idxfp is True:
      vprint('LineIndex:', f'index for {path} already exists')
      continue

    vprint('LineIndex:', f'training on {path}')
    idx = LineIndex()
    idx.train(infp, tqdm=(tqdm if args.progress else None))

    if idxfp is None:
      vprint('LineIndex:', f'not saving index for {path}')
    else:
      idx.dump(idxfp)
      idxfp.close()

    line_indexes[path] = idx
    infp.seek(0)

  for path, idxfp in tqdm(existing_fps.items()):
    vprint('LineIndex:', f'loading existing index for {path}')
    idx = LineIndex()
    idx.load(idxfp)
    line_indexes[path] = idx
    idxfp.close()

  bench_end = time.monotonic()
  vprint('LineIndex:', f'finished in {bench_end - bench_start} seconds')

if args.shuffle_seed == -1:
  args.shuffle_seed = random.randint(1, (2 ** 32) - 1)

bench_start = time.monotonic()
with open(args.output, 'wb') as outfp:
  def write_entry_line(line):
    if len(line.strip()) == 0: return
    entry = json.loads(line)
    for key in args.json_fields:
      if key in entry:
        for l1 in entry[key].split('\n'):
          c = chunk(l1, args.wrap_lines)
          if len(c) == 0: outfp.write(b'\n')
          for l2 in c:
            outfp.write(f'{l2}\n'.encode('utf-8'))

  if args.shuffle_seed == 0:
    for path, infp in zip(args.input_files, input_fps):
      vprint('CorpusWriter:', f'writing {path} without shuffling')
      for line in tqdm(infp):
        try:
          write_entry_line(line)
        except Exception as exc:
          if args.skip_malformed:
            vprint('CorpusWriter:', f'{path}:', exc)
          else:
            eprint('CorpusWriter:', f'{path}:', exc)
            sys.exit(1)
  else:
    for path, infp in zip(args.input_files, input_fps):
      vprint('CorpusWriter:', f'writing {path} with shuffling')
      idx = line_indexes[path]
      subset = list(range(0, idx.total() - 1))

      rand = random.Random(args.shuffle_seed)
      rand.shuffle(subset)
      for lineno in tqdm(subset):
        try:
          line = idx.readline(infp, lineno)
          write_entry_line(line)
        except Exception as exc:
          if args.skip_malformed:
            vprint('CorpusWriter:', f'{path}:', exc)
          else:
            eprint('CorpusWriter:', f'{path}:', exc)
            sys.exit(1)

bench_end = time.monotonic()
vprint('CorpusWriter:', f'finished in {bench_end - bench_start} seconds')
