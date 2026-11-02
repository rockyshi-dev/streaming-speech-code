#!/usr/bin/env bash
# Recreate the three conda environments used by code/tts. The *.lock.txt files beside this
# script are `pip freeze` of the environments as measured; the commands below are the ones
# actually used, and installing from a lock file reproduces the exact versions:
#   pip install -r ssh-tts-cpu.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../tools/roots.sh" && book_conda_init
book_new_env ssh-tts-cpu ssh-tts-gpu ssh-tts-eval   # each env is created only when absent
CACHE_ROOT=$BOOK_DATA/cache
export PIP_CACHE_DIR=$CACHE_ROOT/pip UV_CACHE_DIR=$CACHE_ROOT/uv UV_LINK_MODE=copy

# CPU track: Kokoro, codecs (transformers Mimi/EnCodec/DAC, moshi), Vocos, harness tests
conda create -y -n ssh-tts-cpu --override-channels -c conda-forge "python=$(book_lock_python "$HERE/ssh-tts-cpu.conda.txt")" pip
conda activate ssh-tts-cpu
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install "kokoro==0.9.4" "misaki[en,zh]==0.9.4" soundfile numpy scipy "transformers<5" vocos pytest jiwer moshi==0.2.13
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# GPU track: vLLM-Omni serving Qwen3-TTS on the RTX 4090 (driver 590, CUDA 13.1)
conda create -y -n ssh-tts-gpu --override-channels -c conda-forge "python=$(book_lock_python "$HERE/ssh-tts-gpu.conda.txt")" pip
conda activate ssh-tts-gpu
uv pip install --python "$(which python)" "vllm==0.32.0" "vllm-omni==0.32.0" --torch-backend=auto
uv pip install --python "$(which python)" --reinstall-package torch --reinstall-package torchaudio \
    --reinstall-package torchvision --reinstall-package torchcodec \
    "torch==2.13.0" "torchaudio==2.11.0" "torchvision==0.28.0" "torchcodec==0.17.0" --torch-backend=cu130
# prebuilt flashinfer kernels: the system nvcc (12.4) cannot JIT-compile flashinfer 0.7
uv pip install --python "$(which python)" --no-deps "flashinfer-jit-cache==0.7.0.post1" \
    "flashinfer-jit-cache-sm80==0.7.0.post1" --extra-index-url https://flashinfer.ai/whl/cu130 \
    --index-strategy unsafe-best-match

# Evaluation: Whisper-large-v3 and Paraformer-zh on the GPU
conda create -y -n ssh-tts-eval --override-channels -c conda-forge "python=$(book_lock_python "$HERE/ssh-tts-eval.conda.txt")" pip
conda activate ssh-tts-eval
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.57.6" "funasr==1.4.17" soundfile numpy scipy jiwer pytest modelscope
