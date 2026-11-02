#!/bin/bash
# Build the handbook's figures.
#   figures/chNN/<name>.py    ->  figures/chNN/_out/<name>.svg  (plots, in the bundled style of figures/_fallback)
#   figures/chNN/<name>.tikz  diagrams; they ship as built SVGs in figures/chNN/_out/, which are kept
# BOOK_PY: a Python with matplotlib (default: python3). -f rebuilds plots whose SVG is newer than the source.
# Usage: tools/build-figures.sh [-f] [figures/chNN/<name>.{py,tikz} ...]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD; FORCE=0
[ "${1:-}" = -f ] && { FORCE=1; shift; }
if [ $# -eq 0 ]; then set -- $(ls figures/ch*/*.py figures/ch*/*.tikz 2>/dev/null || true); fi
[ $# -gt 0 ] || { echo "no figure sources found" >&2; exit 1; }
FK=""
# optional external figure tool, used at the tag in figures/FIGKIT_VERSION when it is installed
if [ -z "${NO_FIGKIT:-}" ] && [ -x "$HOME/figkit/bin/figkit" ]; then
  FK="$("$HOME/figkit/bin/figkit" path "$(tr -d '[:space:]' < figures/FIGKIT_VERSION)")/bin/figkit"
  "$FK" lint "$@"
else
  echo "plots in the bundled style; TikZ figures keep their built SVGs" >&2
fi
PY=${BOOK_PY:-python3}
"$PY" -c "import matplotlib" 2>/dev/null || { echo "$PY has no matplotlib: set BOOK_PY to a Python that has it" >&2; exit 1; }
for SRC in "$@"; do
  DIR=$(dirname "$SRC")/_out; NAME=$(basename "${SRC%.*}"); OUT="$DIR/$NAME.svg"; mkdir -p "$DIR"
  if [ $FORCE = 0 ] && [ -f "$OUT" ] && [ -z "$(find "$SRC" figures/bookstyle.py figures/FIGKIT_VERSION -newer "$OUT" | head -1)" ]; then
    echo "up to date: $OUT"; continue
  fi
  case "$SRC" in
    *.py) if [ -n "$FK" ]; then FIGKIT_PY=$PY PYTHONPATH="$REPO" "$FK" plot -t mono-accent "$SRC" "$OUT"
          else FIGKIT_OUT="$REPO/$OUT" PYTHONPATH="$REPO" "$PY" "$SRC"; fi ;;
    *.tikz) [ -n "$FK" ] || { echo "kept: $OUT (built SVG)"; continue; }
            "$FK" tikz -t mono-accent "$SRC" "$OUT" ;;
  esac
  echo "built: $OUT"
done
