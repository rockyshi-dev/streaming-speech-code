#!/usr/bin/env bash
# Create the two conda environments of code/serve from the pip freezes beside this script (Linux x86-64).
#   ssh-serve  sherpa-onnx server, load generators, transport, cascade client (CPU); opuslib needs the
#              system libopus (for example `apt install libopus0`)
#   ssh-serve-nemo  ch15/nemo_server.py: NeMo (git a3afbd9), torch 2.11 cu126. websockets 17.3 is installed into
#              $BOOK_DATA/pylib-nemo and put on PYTHONPATH by the scripts that start the server.
# Each env is created only when no env of that name exists. Python is pinned to the version in the lock file.
# Run from anywhere: code/serve/env/create_envs.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../tools/roots.sh" && book_conda_init
book_new_env ssh-serve ssh-serve-nemo

conda create -y -n ssh-serve --override-channels -c conda-forge "python=$(book_lock_python "$HERE/ssh-serve.lock.txt")" pip
conda activate ssh-serve
pip install -r "$HERE/ssh-serve.lock.txt"

conda create -y -n ssh-serve-nemo --override-channels -c conda-forge \
    "python=$(book_lock_python "$HERE/ssh-serve-nemo.lock.txt")" pip
conda activate ssh-serve-nemo
pip install -r "$HERE/ssh-serve-nemo.lock.txt" --extra-index-url https://download.pytorch.org/whl/cu126
pip install --target "$BOOK_DATA/pylib-nemo" websockets==17.3
