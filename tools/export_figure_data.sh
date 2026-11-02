#!/bin/bash
# Regenerate every figure data file (figures/chNN/data/*.json, *.npz) from the result files, in dependency
# order, then rebuild the figures. Run after the measurements; code/ch09_measure/make_tables.py must have run
# first, because the ch05 and ch09 exporters read its summary table.
#   tools/export_figure_data.sh        # BOOK_PY: a Python with numpy and matplotlib (default: the ssh-asr env)
set -euo pipefail
cd "$(dirname "$0")/.."
source code/tools/roots.sh
PY=${BOOK_PY:-$(book_python ssh-asr)}
for f in figures/ch*/data/export_*.py; do
  echo "== $f"
  "$PY" "$f"
done
BOOK_PY=$PY tools/build-figures.sh -f
