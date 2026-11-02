#!/usr/bin/env bash
# Write a copy of the installed vLLM-Omni Qwen3-TTS deploy configuration in which both stages' `max_num_seqs`
# is NSEQS (default 192; the package ships 64), and print the path of the copy. Nothing else in the file changes:
# the script stops unless the original has exactly two `max_num_seqs: 64` lines and a diff against the copy shows
# exactly those two lines. Pass the path to the server as DEPLOY_CONFIG (see with_server.sh).
#   code/tts/gpu/make_deploy_config.sh [NSEQS]
set -euo pipefail
NSEQS=${1:-192}
[[ "$NSEQS" =~ ^[0-9]+$ ]] || { echo "NSEQS must be an integer" >&2; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../tools/roots.sh" && book_activate ssh-tts-gpu
PKG=$(python -c 'import vllm_omni, os; print(os.path.dirname(vllm_omni.__file__))')
SRC=$PKG/deploy/qwen3_tts.yaml
DST_DIR=$BOOK_DATA/runs/deploy
DST=$DST_DIR/qwen3_tts_seqs$NSEQS.yaml
mkdir -p "$DST_DIR"

n=$(grep -c '^    max_num_seqs: 64$' "$SRC" || true)
[[ "$n" == 2 ]] || { echo "expected exactly two '    max_num_seqs: 64' lines in $SRC, found $n" >&2; exit 2; }
sed "s/^    max_num_seqs: 64\$/    max_num_seqs: $NSEQS/" "$SRC" > "$DST"

removed=$(diff "$SRC" "$DST" | grep -c '^<' || true)
added=$(diff "$SRC" "$DST" | grep -c '^>' || true)
[[ "$removed" == 2 && "$added" == 2 ]] || { echo "diff is not exactly two lines ($removed removed, $added added)" >&2; exit 3; }
grep -c "^    max_num_seqs: $NSEQS\$" "$DST" | grep -qx 2 || { echo "the copy does not hold two max_num_seqs: $NSEQS lines" >&2; exit 4; }
echo "$DST"
