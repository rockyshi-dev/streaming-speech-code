#!/usr/bin/env bash
# The chapter 15 GPU sweeps: Nemotron 3.5 ASR streaming at its trained chunk sizes (80, 320, 560 ms), fp32,
# plus bf16 at 320 ms, each configuration once at all its stream counts, one result file per configuration
# (results/ch15_nemo_gpu_<name>.json). One GPU, about 50 minutes:
#   code/serve/ch15/run_gpu_sweeps.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source ../tools/roots.sh
P=$(book_python ssh-serve) || exit 1
N=$(book_python ssh-serve-nemo) || exit 1
R=$BOOK_DATA/runs
NEMO="env PYTHONPATH=$BOOK_DATA/pylib-nemo $N ch15/nemo_server.py --port 6010"
run() {  # name chunk_ms r dtype stop_ms maxbatch streams...
    local name=$1 ms=$2 r=$3 dt=$4 stop=$5 mb=$6; shift 6
    mkdir -p $R/$name
    $P ch15/sweep.py --name $name --port 6010 --lang-hint --chunk-ms $ms --stop-p95-ms $stop --streams "$@" \
        --duration 60 -- $NEMO --context-right $r --dtype $dt --max-batch $mb --step-log $R/$name/steps.jsonl
}
run nemo_gpu_r3_fp32 320 3 fp32 1500 512 1 8 32 64 128 192 256 320 384
run nemo_gpu_r0_fp32  80 0 fp32 1000 256 1 8 16 32 48 64 96 128
run nemo_gpu_r6_fp32 560 6 fp32 2000 512 1 32 128 256 384 512
run nemo_gpu_r3_bf16 320 3 bf16 1500 512 1 32 128 192 256 320 384 448
