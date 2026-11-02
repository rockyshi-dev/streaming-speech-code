#!/usr/bin/env bash
# Recreate the two environments from the lock files (Linux x86-64).
#   ssh-asr   CPU track: numpy/torch CPU, torchaudio, sherpa-onnx, silero-vad, kaldi-native-fbank, jiwer
#   ssh-nemo  GPU track: torch 2.8.0+cu126, NVIDIA NeMo (ASR collection)
# conda-forge only (no defaults channel), python 3.11.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../tools/roots.sh" && book_conda_init
book_new_env ssh-asr ssh-nemo   # each env is created only when absent; the explicit files pin Python
for e in ssh-asr ssh-nemo; do
  conda create -y -n "$e" --file "$HERE/$e.conda.explicit.txt"
done
conda activate ssh-asr
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchaudio==2.8.0+cpu
pip install -r "$HERE/ssh-asr.requirements.lock.txt" --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e "$HERE/.."
conda activate ssh-nemo
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.8.0+cu126 torchaudio==2.8.0+cu126
pip install -r "$HERE/ssh-nemo.requirements.lock.txt" --extra-index-url https://download.pytorch.org/whl/cu126
pip install -e "$HERE/.."
