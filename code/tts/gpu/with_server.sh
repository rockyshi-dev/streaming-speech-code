#!/usr/bin/env bash
# Start a vLLM-Omni Qwen3-TTS server on the local GPU, wait until it answers, run one client
# command against it, then stop the server. For example, from the repository root:
#
#   code/tts/gpu/with_server.sh <log-dir> python code/tts/gpu/qwen3tts_client.py smoke
#
# Environment overrides: BOOK_TTS_MODEL (Hub id), DEPLOY_CONFIG (path; default = the package's qwen3_tts.yaml),
# BOOK_TTS_PORT (default 8091). The values used are written to <log-dir>/server_cmd.txt.
set -euo pipefail

LOGDIR=$1; shift
mkdir -p "$LOGDIR"

source "$(dirname "$0")/../../tools/roots.sh" && book_activate ssh-tts-gpu

MODEL=${BOOK_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}
PORT=${BOOK_TTS_PORT:-8091}
PKG=$(python -c 'import vllm_omni, os; print(os.path.dirname(vllm_omni.__file__))')
DEPLOY_CONFIG=${DEPLOY_CONFIG:-$PKG/deploy/qwen3_tts.yaml}

# keep every cache off $HOME
CACHE=$BOOK_DATA/cache
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer
export HF_HUB_OFFLINE=1 SPEAKER_SAMPLES_DIR=$CACHE/vllm-omni-speakers  # default is ~/.cache/vllm-omni/speakers
mkdir -p "$SPEAKER_SAMPLES_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

if curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null || \
   python -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1', $PORT)) != 0)"; then
    echo "port $PORT is already in use; set BOOK_TTS_PORT to a free port" >&2; exit 4
fi
echo "model=$MODEL deploy=$DEPLOY_CONFIG port=$PORT" | tee "$LOGDIR/server_cmd.txt"
setsid vllm serve "$MODEL" --omni --deploy-config "$DEPLOY_CONFIG" --port "$PORT" \
    --stage-init-timeout 1200 --init-timeout 1800 >"$LOGDIR/server.log" 2>&1 &
SPID=$!
cleanup() {
    if kill -0 "$SPID" 2>/dev/null; then
        kill -TERM -- -"$SPID" 2>/dev/null || true
        for _ in $(seq 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
        kill -KILL -- -"$SPID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$SPID" 2>/dev/null; then echo "server exited during start-up; see $LOGDIR/server.log" >&2; exit 2; fi
    if (( $(date +%s) - t0 > 1800 )); then echo "server not ready after 30 min" >&2; exit 3; fi
    # the API server announces its port before it accepts connections; 3 min after that is a hang
    if grep -q "Starting vLLM API server" "$LOGDIR/server.log" 2>/dev/null; then
        t_api=${t_api:-$(date +%s)}
        if (( $(date +%s) - t_api > 180 )); then
            echo "API server announced but /health not answering after 180 s; stopping" >&2; exit 5; fi
    fi
    sleep 5
done
echo "server ready after $(( $(date +%s) - t0 )) s" | tee -a "$LOGDIR/server_cmd.txt"

export TTS_URL="http://127.0.0.1:$PORT" BOOK_TTS_MODEL="$MODEL"
"$@"
