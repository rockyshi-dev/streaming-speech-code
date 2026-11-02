#!/usr/bin/env bash
# GPU track, chapters 5 and 9: every GPU step, one after the other, on one GPU (about an hour on an RTX 4090):
#   code/ch09_measure/run_gpu_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/roots.sh && book_activate ssh-nemo
# 0. smoke: 10 utterances, two contexts (fails fast on a defect)
python ch09_measure/eval_nemo.py --manifests librispeech-test-other-200 --decoders rnnt --contexts 70,13 70,0 \
  --limit 10 --results-dir "$BOOK_DATA/runs/smoke"
# 1. adapter check for Nemotron 3.5 (FastConformer was checked by check_adapter.py already)
python ch05_streaming_gpu/check_adapter.py --model nemotron-3.5 --manifest aishell1-test-200 --target-lang zh-CN --n 10
# 2. FastConformer multi-latency sweep: 4 contexts x {transducer, CTC} x {test-clean, test-other}
python ch09_measure/eval_nemo.py --model fastconformer-en-multi \
  --manifests librispeech-test-clean-200 librispeech-test-other-200 --decoders rnnt ctc
# 3. Nemotron 3.5 on AISHELL-1 (Mandarin prompt), 4 contexts
python ch09_measure/eval_nemo.py --model nemotron-3.5 --manifests aishell1-test-200 --decoders rnnt --target-lang zh-CN
