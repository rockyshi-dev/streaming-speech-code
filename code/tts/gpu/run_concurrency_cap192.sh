#!/usr/bin/env bash
# Chapter 14 follow-up: the concurrency sweep with the server configured at max_num_seqs 192 in both stages
# (in short: where does the server overload when its
# own cap of 64 is out of the way). One GPU, about 12 minutes. Run from the repository root:
#   code/tts/gpu/run_concurrency_cap192.sh
# -> results/ch14_qwen3tts_concurrency_cap192.json, plus $BOOK_DATA/runs/ch14_cap192_gpu_samples.csv (SM clock,
#    power and memory once per second for the whole run) and the server log under runs/qwen3tts_server_cap192/.
# Level 64 is listed twice on purpose: the first visit is the control, the second (after 192, same server) measures drift.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../tools/roots.sh"
export DEPLOY_CONFIG
DEPLOY_CONFIG=$("$HERE/make_deploy_config.sh" 192)
echo "deploy config: $DEPLOY_CONFIG"
SAMPLES=$BOOK_DATA/runs/ch14_cap192_gpu_samples.csv
mkdir -p "$(dirname "$SAMPLES")"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu,temperature.gpu \
    --format=csv -l 1 > "$SAMPLES" &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT
"$HERE/with_server.sh" "$BOOK_DATA/runs/qwen3tts_server_cap192" \
    python -u "$HERE/qwen3tts_client.py" concurrency --name ch14_qwen3tts_concurrency_cap192 \
    --levels 64,80,96,112,128,160,192,64 --requests 40 --min-rounds 3
