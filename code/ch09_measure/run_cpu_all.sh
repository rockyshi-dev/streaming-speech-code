#!/usr/bin/env bash
# Chapter 6 and 9 CPU-track runs: greedy and modified beam search on the three fixed subsets.
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/roots.sh && book_activate ssh-asr
for dec in greedy_search modified_beam_search; do
  for m in librispeech-test-clean-200 librispeech-test-other-200 aishell1-test-200; do
    python ch09_measure/eval_sherpa.py --manifest $m --decoding $dec
  done
done
