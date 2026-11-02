#!/usr/bin/env bash
# Chapter 14 concurrency sweep (fresh server), then the ASR intelligibility of every set, on one GPU
# (about 40 minutes). Run from the repository root.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
"$HERE/run_concurrency_gpu.sh"
"$HERE/../ch14_harness/run_eval_gpu.sh"
