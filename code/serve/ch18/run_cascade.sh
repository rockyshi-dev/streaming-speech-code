#!/usr/bin/env bash
# Start the three GPU servers of the chapter 18 cascade on one RTX 4090, wait until each answers, run the
# given commands (each a cascade.py invocation, run in code/serve), then stop everything. Start it in the
# ssh-serve environment; one GPU with 24 GB:
#
#   code/serve/ch18/run_cascade.sh <log-dir> "python ch18/cascade.py --name cascade_smoke --turns 4" ["..."]
#
# Servers (all on 127.0.0.1):
#   8091  vLLM-Omni 0.32.0, Qwen3-TTS-12Hz-1.7B-CustomVoice, deploy config ch18/qwen3_tts_cascade.yaml
#   8000  vLLM 0.32.0, Qwen3-1.7B (bf16), 0.267 of GPU memory, max length 2048, served as "qwen3-1.7b"
#   6010  ch15/nemo_server.py, Nemotron 3.5 ASR streaming 0.6B, 320 ms chunks ([56,3], trained), fp32
set -euo pipefail
LOGDIR=$1; shift
mkdir -p "$LOGDIR"
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/../tools/roots.sh" && book_conda_init
NEMO_PY=$(book_python ssh-serve-nemo) || exit 1

CACHE=$BOOK_DATA/cache
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer
export HF_HUB_OFFLINE=1
mkdir -p "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

for port in 8091 8000 6010; do  # never time someone else's server that happens to hold our port
    if (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null; then echo "port $port already in use; refusing to start" >&2; exit 4; fi
done

PIDS=()
cleanup() {
    for p in "${PIDS[@]}"; do kill -TERM -- -"$p" 2>/dev/null || true; done
    for p in "${PIDS[@]}"; do for _ in $(seq 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done; kill -KILL -- -"$p" 2>/dev/null || true; done
}
trap cleanup EXIT

wait_http() {  # url pid name
    local t0=$(date +%s)
    until curl -sf "$1" >/dev/null 2>&1; do
        kill -0 "$2" 2>/dev/null || { echo "$3 exited during start-up" >&2; exit 2; }
        (( $(date +%s) - t0 > 1500 )) && { echo "$3 not ready after 25 min" >&2; exit 3; }
        sleep 3
    done
    echo "$3 ready after $(( $(date +%s) - t0 )) s" | tee -a "$LOGDIR/servers.txt"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/  GPU memory after $3: /" | tee -a "$LOGDIR/servers.txt"
}

conda activate ssh-tts-gpu || { echo "conda activate ssh-tts-gpu failed" >&2; exit 1; }
setsid vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --deploy-config "$HERE/ch18/qwen3_tts_cascade.yaml" \
    --port 8091 --stage-init-timeout 1200 --init-timeout 1800 >"$LOGDIR/tts.log" 2>&1 &
PIDS+=($!); wait_http http://127.0.0.1:8091/health ${PIDS[-1]} tts

setsid vllm serve "$BOOK_MODELS/Qwen3-1.7B" --served-model-name qwen3-1.7b --port 8000 \
    --gpu-memory-utilization 0.267 --max-model-len 2048 --max-num-seqs 8 >"$LOGDIR/llm.log" 2>&1 &
PIDS+=($!); wait_http http://127.0.0.1:8000/health ${PIDS[-1]} llm
conda deactivate

PYTHONPATH="$BOOK_DATA/pylib-nemo" setsid "$NEMO_PY" \
    "$HERE/ch15/nemo_server.py" --port 6010 --context-right 3 --dtype fp32 >"$LOGDIR/asr.log" 2>&1 &
PIDS+=($!)
t0=$(date +%s)
until grep -q READY "$LOGDIR/asr.log"; do
    kill -0 ${PIDS[-1]} 2>/dev/null || { echo "asr exited during start-up" >&2; exit 2; }
    (( $(date +%s) - t0 > 600 )) && { echo "asr not ready" >&2; exit 3; }
    sleep 2
done
echo "asr ready after $(( $(date +%s) - t0 )) s" | tee -a "$LOGDIR/servers.txt"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/  GPU memory after asr: /" | tee -a "$LOGDIR/servers.txt"

cd "$HERE"
for cmd in "$@"; do
    echo ">>> $cmd" | tee -a "$LOGDIR/servers.txt"
    bash -c "$cmd"
done
