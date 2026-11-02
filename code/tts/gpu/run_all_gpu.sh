#!/usr/bin/env bash
# Chapter 12: one server start, then the input-mode run (results/ch12_qwen3tts_input_modes.json), on one GPU.
# Run from the repository root:
#   code/tts/gpu/run_all_gpu.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../tools/roots.sh"
LOG=$BOOK_DATA/runs/qwen3tts_server_measure
exec "$HERE/with_server.sh" "$LOG" bash -c "
  python -u '$HERE/qwen3tts_client.py' input-modes --rate 20 --repeats 2"
