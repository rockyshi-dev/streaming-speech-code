#!/usr/bin/env bash
# ASR intelligibility of every synthesized set, on one GPU (about 25 minutes). Run from the repository root.
set -euo pipefail
source "$(dirname "$0")/../../tools/roots.sh" && book_activate ssh-tts-eval
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-$BOOK_DATA/cache/modelscope} HF_HUB_OFFLINE=1
R=$BOOK_DATA/runs
mkdir -p $R/ch14_intelligibility
python -u "$(dirname "$0")/eval_intelligibility.py" --name ch14_intelligibility \
  --source human \
  --source kokoro:$R/ch12_kokoro_cpu/rows.jsonl \
  --source qwen-modes:$R/ch12_qwen3tts_input_modes/rows.jsonl \
  --source qwen-conc:$R/ch14_qwen3tts_concurrency/rows.jsonl
