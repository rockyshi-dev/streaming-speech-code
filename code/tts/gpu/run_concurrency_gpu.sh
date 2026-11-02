#!/usr/bin/env bash
# Concurrency sweep only (chapter 14), with a fresh server, on one GPU (about 15 minutes). Run from the repository root.
# BOOK_TTS_LEVELS overrides the comma-separated concurrency levels.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../tools/roots.sh"
exec "$HERE/with_server.sh" "$BOOK_DATA/runs/qwen3tts_server_concurrency" \
  python -u "$HERE/qwen3tts_client.py" concurrency --levels "${BOOK_TTS_LEVELS:-1,2,4,8,16,24,32,48,64}" \
  --requests 40 --min-rounds 3 --save-audio
