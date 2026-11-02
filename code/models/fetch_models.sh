#!/usr/bin/env bash
# Fetch the pre-trained models used by code/ into $BOOK_MODELS, at pinned revisions.
# Re-runnable: a file is present when it has the expected size (given after each name); present files and
# directories are left untouched.
#   code/models/fetch_models.sh            # chapters 1-9 (about 1.1 GB)
#   code/models/fetch_models.sh --serve    # also chapters 15-18 (about 4 GB more, mostly Qwen3-1.7B)
set -euo pipefail
source "$(dirname "$0")/../tools/roots.sh"
STORE=$BOOK_MODELS
HF=${HF_ENDPOINT:-https://huggingface.co}

size_of() { stat -L -c %s "$1" 2>/dev/null || echo -1; }

fetch() {  # repo revision local-name file=bytes...
  # A file is present when it exists with the expected size. A directory whose files are all present is not
  # touched at all; otherwise only the missing or wrong files are downloaded, each checked for its size, and
  # SOURCE.json is written only if the directory has none.
  local repo=$1 rev=$2 name=$3; shift 3
  local dir="$STORE/$name" f want todo=()
  for spec in "$@"; do
    f=${spec%=*}; want=${spec##*=}
    [[ "$(size_of "$dir/$f")" == "$want" ]] || todo+=("$spec")
  done
  if (( ${#todo[@]} == 0 )); then echo "present: $name"; return 0; fi
  for spec in "${todo[@]}"; do
    f=${spec%=*}; want=${spec##*=}
    mkdir -p "$dir/$(dirname "$f")"
    echo "fetch $repo@$rev $f"
    curl -fsSL --retry 5 -o "$dir/$f.part" "$HF/$repo/resolve/$rev/$f"
    [[ "$(size_of "$dir/$f.part")" == "$want" ]] || { echo "$f: expected $want bytes, got $(size_of "$dir/$f.part")" >&2; exit 1; }
    mv "$dir/$f.part" "$dir/$f"
  done
  [[ -e "$dir/SOURCE.json" ]] || printf '{"repo": "%s", "revision": "%s", "files": %d}\n' "$repo" "$rev" "$#" > "$dir/SOURCE.json"
}

# English streaming Zipformer transducer (icefall, LibriSpeech 960 h), Apache-2.0
fetch csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26 672fbf1b30579d6585301139bb363f42a0ad4a24 \
  sherpa-onnx-streaming-zipformer-en-2023-06-26 \
  README.md=216 bpe.model=244865 tokens.txt=5048 test_wavs/0.wav=212044 test_wavs/1.wav=534924 \
  test_wavs/trans.txt=449 encoder-epoch-99-avg-1-chunk-16-left-128.onnx=262127043 \
  decoder-epoch-99-avg-1-chunk-16-left-128.onnx=2092621 joiner-epoch-99-avg-1-chunk-16-left-128.onnx=1026405 \
  encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx=71083163 \
  decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx=1307236 \
  joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx=259335

# Mandarin streaming Zipformer transducer (icefall multi_zh-hans), Apache-2.0
fetch k2-fsa/sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12 ac54a23c9d106dfbd178be831329fabe261bac58 \
  sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12 \
  README.md=28 bpe.model=263956 tokens.txt=18626 test_wavs/DEV_T0000000000.wav=179646 \
  encoder-epoch-20-avg-1-chunk-16-left-128.onnx=260643233 decoder-epoch-20-avg-1-chunk-16-left-128.onnx=5165083 \
  joiner-epoch-20-avg-1-chunk-16-left-128.onnx=4104464 \
  encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx=70109350 \
  decoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx=1308688 \
  joiner-epoch-20-avg-1-chunk-16-left-128.int8.onnx=1033416

# English cache-aware FastConformer hybrid transducer/CTC, multi-latency, CC-BY-4.0
fetch nvidia/stt_en_fastconformer_hybrid_large_streaming_multi ae98143333690bd7ced4bc8ec16769bcb8918374 \
  stt_en_fastconformer_hybrid_large_streaming_multi \
  README.md=10175 stt_en_fastconformer_hybrid_large_streaming_multi.nemo=459673600

# Nemotron 3.5 ASR streaming 0.6B (OpenMDW-1.1) is read from the Hugging Face cache
# ($HF_HOME/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b, revision ea30d66d), not copied.

[[ "${1:-}" == --serve ]] || exit 0

# Chapters 15-18. Bilingual zh-en streaming Zipformer (the chapter 15 CPU server), Apache-2.0
fetch csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 98590b7ed6443e77b714204da2757d75e1a642f4 \
  sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 \
  README.md=296 bpe.model=244836 tokens.txt=56317 encoder-epoch-99-avg-1.int8.onnx=181895032 \
  decoder-epoch-99-avg-1.onnx=13876452 joiner-epoch-99-avg-1.int8.onnx=3228404

# Smart Turn v3.2 CPU (the chapter 18 turn detector), BSD-2-Clause
fetch pipecat-ai/smart-turn-v3 f766f81d3cfdf7737ac64aad813d91bbfd56bf93 smart-turn-v3 \
  README.md=1247 smart-turn-v3.2-cpu.onnx=8679182

# Qwen3-1.7B (the chapter 18 language model, served by vLLM), Apache-2.0
fetch Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e Qwen3-1.7B \
  config.json=726 generation_config.json=239 merges.txt=1671853 vocab.json=2776833 tokenizer.json=11422654 \
  tokenizer_config.json=9732 model.safetensors.index.json=25605 model-00001-of-00002.safetensors=3441185608 \
  model-00002-of-00002.safetensors=622329984
