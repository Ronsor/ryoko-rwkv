#!/bin/sh
# Copyright (C) Ronsor Labs and the Ryoko AI Production Committee.
#
# This software may be used and distributed according to the terms of the
# Apache License version 2.0, a copy of which may be found in the LICENSE file
# at the root of this repository.

failure() {
  printf '%s: %s\n' \
         "$1" "Failed to launch SentencePiece trainer. Is it installed?" >&2
}

[ -z "$SPM_TRAIN" ] && SPM_TRAIN="spm_train"
if ! command -v "$SPM_TRAIN" 2>&1 >/dev/null; then
  failure "$0"
  exit 1
fi

# Standard control symbols
# - Dataset metadata symbols
CONTROL_SYMBOLS="<|meta|>,<|title|>,<|lang|>,<|subset|>,<|tag|>"
# - Control signal symbols
CONTROL_SYMBOLS="$CONTROL_SYMBOLS<|exec|>,<|system|>,<|prompt|>,<|reply|>,"
# - Misc symbols
CONTROL_SYMBOLS="$CONTROL_SYMBOLS<|A|>,<|B|>,<|C|>,<|W|>,<|X|>,<|Y|>,<|Z|>"

# These parameters are designed to imitate the tokenizer used by the LLaMA
# large language model.
exec "$SPM_TRAIN" \
  --model_type="bpe"              \
  --character_coverage=0.99995    \
  --max_sentence_length=4192      \
  --max_sentencepiece_length=16   \
  --split_by_unicode_script=true  \
  --split_by_number=true          \
  --split_digits=true             \
  --byte_fallback=true            \
  --hard_vocab_limit=true         \
  --shuffle_input_sentence=true   \
  --pad_id=-1                     \
  --allow_whitespace_only_pieces=true  \
  --control_symbols="$CONTROL_SYMBOLS" \
  "$@"

failure "$0"
exit 2
