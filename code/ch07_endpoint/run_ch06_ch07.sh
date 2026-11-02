#!/usr/bin/env bash
# CPU: chapter 7 endpoint sweep, then chapter 6 hotword runs (needs run_cpu_all.sh results first).
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/roots.sh && book_activate ssh-asr
python ch07_endpoint/endpoint_sweep.py --n 100
python ch06_decoding/hotwords.py --manifest librispeech-test-other-200
python ch06_decoding/hotwords.py --manifest aishell1-test-200
