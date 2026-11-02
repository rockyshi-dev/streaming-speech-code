# Serving and real-time voice (chapters 15-18)

Code and measurement harnesses for Part IV (serving streaming ASR and TTS) and Part V (transport, the voice
cascade). Everything runs on one machine: Intel i5-12600K (16 threads) and one RTX 4090 (23,028 MiB usable). Results
are summary JSON files in `results/`; raw per-session and per-turn records are under
`$BOOK_DATA/runs/<name>/` (the three directories `BOOK_DATA`, `BOOK_DATASETS` and `BOOK_MODELS` are described in
`../README.md`, "Setting up"). The commands behind every number are under "How to run" below; the numbers
themselves are in `results/` and in chapters 15-18.

## Layout

| Path | What |
|---|---|
| `common/util.py` | audio loading, percentiles, WER/CER, run metadata |
| `common/stream_features.py` | incremental log-mel that equals the offline front end frame for frame |
| `common/turn.py` | Silero VAD (ONNX, no torch), Smart Turn v3.2, the endpointer logic |
| `common/handoff.py` | LLM-to-TTS sentence aggregation |
| `common/player.py` | virtual audio player: when each sample is heard, underruns, stop for barge-in |
| `common/build_manifest.py` | the fixed 40 + 40 test utterances (`data/test_utts.jsonl`) |
| `ch15/sherpa_server.py` | sherpa-onnx's streaming WebSocket server (v2.0.1), plus two timing fields |
| `ch15/nemo_server.py` | GPU server: Nemotron 3.5 ASR streaming, per-stream caches, batched step loop |
| `ch15/loadgen.py` | N concurrent real-time streams; partial and final latency |
| `ch15/sweep.py` | start a server, run the load generator at each N, collect |
| `ch16/tts_openloop.py` | open-loop (Poisson) and streamed-text load on the chapter 14 TTS server |
| `ch17/opus_delay.py` | Opus frame sizes, look-ahead, measured codec delay, PLC vs FEC |
| `ch17/jitter.py` | toy jitter buffer under synthetic jitter; UDP-like vs TCP-like delivery |
| `ch17/ws_transport.py` | file-at-real-time client and receiver over WebSocket, 20 ms frames, PCM or Opus |
| `ch18/cascade.py` | VAD + turn detector -> streaming ASR -> LLM -> streaming TTS, timed per turn; barge-in |
| `ch18/run_cascade.sh` | starts the three GPU servers on the 4090, runs cascade commands, stops them |
| `ch18/qwen3_tts_cascade.yaml` | vLLM-Omni Qwen3-TTS deploy config with reduced memory for sharing the card |
| `tests/` | CPU unit tests (`pytest -q code/serve/tests`) |
| `env/` | `create_envs.sh` and the pip freezes of the environments used |

## Environments

| Env | Used for | Lock |
|---|---|---|
| `ssh-serve` (conda-forge Python 3.11) | sherpa-onnx server, load generators, transport, cascade client | `env/ssh-serve.lock.txt` |
| `ssh-serve-nemo` (NeMo 3.1.0, torch 2.11 cu126) + `websockets` 17.3 installed with `pip --target $BOOK_DATA/pylib-nemo` and put on `PYTHONPATH` | `ch15/nemo_server.py` | `env/ssh-serve-nemo.lock.txt` |
| `ssh-tts-gpu` (the env of `code/tts`; vLLM 0.32.0, vLLM-Omni 0.32.0) | Qwen3-TTS server and the Qwen3-1.7B LLM server | see `code/tts` |

`env/create_envs.sh` builds `ssh-serve` and `ssh-serve-nemo` from their locks, including the `websockets` install into
`$BOOK_DATA/pylib-nemo`. `ssh-serve-nemo` is a different environment from `ssh-nemo` of chapters 1-9 (NeMo at git
commit a3afbd9 and torch 2.11 here, NeMo 3.0.0 and torch 2.8 there).

## Models and licences

| Model | Where | Licence |
|---|---|---|
| sherpa-onnx streaming Zipformer bilingual zh-en 2023-02-20 (`csukuangfj/...@98590b7`) | `$BOOK_MODELS/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` | Apache-2.0 |
| Nemotron 3.5 ASR streaming 0.6B (`nvidia/nemotron-3.5-asr-streaming-0.6b@ea30d66`) | HF cache | OpenMDW-1.1 |
| Silero VAD v6 (`silero-vad` 6.2.3 wheel, `silero_vad.onnx` sha256 `1a153a22...`) | `$BOOK_DATA/vendor/silero` (see below) | MIT |
| Smart Turn v3.2 CPU (`pipecat-ai/smart-turn-v3@f766f81`) | `$BOOK_MODELS/smart-turn-v3` | BSD-2-Clause |
| Qwen3-1.7B (`Qwen/Qwen3-1.7B@70d244c`) | `$BOOK_MODELS/Qwen3-1.7B` | Apache-2.0 |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | HF cache (see `code/tts`) | Apache-2.0 |
| libopus 1.3.1 (system) via opuslib 3.0.1 | | BSD-3-Clause |

`code/models/fetch_models.sh --serve` fetches the bilingual Zipformer, Smart Turn and Qwen3-1.7B into
`$BOOK_MODELS` at the revisions above. Nemotron 3.5 is fetched into the Hugging Face cache as `../README.md`
("Setting up") shows, Qwen3-TTS as `code/tts/README.md` shows. Silero VAD is the ONNX file inside the
`silero-vad` wheel, unpacked into `$BOOK_DATA/vendor/silero` so that
`$BOOK_DATA/vendor/silero/silero_vad/data/silero_vad.onnx` exists:

```bash
pip download --no-deps silero-vad==6.2.3 -d "$BOOK_DATA/vendor/silero"
unzip -o "$BOOK_DATA/vendor/silero/silero_vad-6.2.3-py3-none-any.whl" -d "$BOOK_DATA/vendor/silero"
```

Code fetched from upstream at pinned versions (not committed here): sherpa-onnx
`python-api-examples/streaming_server.py` and `http_server.py` at tag v2.0.1 (Apache-2.0; sha256
`38f989bf...` and `a78a7df7...`) into `$BOOK_DATA/vendor/sherpa-onnx-v2.0.1/`;
pipecat-ai/smart-turn `inference.py` / `audio_utils.py` at commit `4786657` (BSD-2-Clause, reference for the
preprocessing in `common/turn.py`).

Test audio: LibriSpeech test-clean (CC BY 4.0) and AISHELL-1 test (Apache-2.0), read in place from
`$BOOK_DATASETS`; the 80 utterances are listed in `data/test_utts.jsonl`.

## How to run

The commands of every measurement in chapters 15 to 18, run from `code/serve`. `<logdir>` is any directory for
server logs. Each command names the result file it writes in `results/`.

```bash
cd code/serve; P=python   # in the ssh-serve environment
ch15/fetch_vendor.sh      # once: sherpa-onnx's server files, checked by sha256
M=$BOOK_MODELS/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
MODEL="--encoder $M/encoder-epoch-99-avg-1.int8.onnx --decoder $M/decoder-epoch-99-avg-1.onnx --joiner $M/joiner-epoch-99-avg-1.int8.onnx --tokens $M/tokens.txt"

# ch15 CPU: the vendor defaults (batch 3, 2 threads) -> ch15_sherpa_cpu_b3t2.json
$P ch15/sweep.py --name sherpa_cpu_b3t2 --streams 1 2 4 8 16 32 48 64 --stop-p95-ms 1500 --duration 60 -- \
   $P ch15/sherpa_server.py --port 6006 --use-endpoint 0 --max-batch-size 3 --num-threads 2 --nn-pool-size 1 $MODEL
# ch15 CPU: a tuned server (batch 16, 4 threads, 2 decode threads) -> ch15_sherpa_cpu_b16t4p2.json
$P ch15/sweep.py --name sherpa_cpu_b16t4p2 --streams 1 8 16 32 48 64 96 --stop-p95-ms 1500 --duration 60 -- \
   $P ch15/sherpa_server.py --port 6006 --use-endpoint 0 --max-batch-size 16 --num-threads 4 --nn-pool-size 2 $MODEL

# ch15 GPU (one GPU, about 50 minutes): 80/320/560 ms fp32 and 320 ms bf16, one result file per configuration
# -> ch15_nemo_gpu_r0_fp32.json, ch15_nemo_gpu_r3_fp32.json, ch15_nemo_gpu_r6_fp32.json, ch15_nemo_gpu_r3_bf16.json
ch15/run_gpu_sweeps.sh

# ch16 (one GPU, about 35 minutes), inside the TTS server wrapper of code/tts/gpu with its default deploy config
# -> ch16_openloop.json and ch16_wsload.json; then, with a fresh server, ch16_openloop_high.json
../tts/gpu/with_server.sh <logdir> bash -c \
  "python ch16/tts_openloop.py openloop --rates 0.5,1,2,4,6,8 --requests 50 && \
   python ch16/tts_openloop.py wsload --levels 1,2,4,8,16 --requests 48 --text-rate 20"
../tts/gpu/with_server.sh <logdir> \
  python ch16/tts_openloop.py openloop --name ch16_openloop_high --rates 10,12,16,20 --requests 80 --seed 1

# ch17 (CPU) -> ch17_opus_delay.json, ch17_jitter_synthetic.json, ch17_ws_pcm.json, ch17_ws_opus.json,
#    ch17_ws_opus_jitter_wifi.json, ch17_ws_opus_jitter_mobile.json
$P ch17/opus_delay.py
$P ch17/jitter.py
$P ch17/ws_transport.py run --codec pcm --out $BOOK_DATA/runs/ch17_ws/pcm
$P ch17/ws_transport.py run --codec opus --out $BOOK_DATA/runs/ch17_ws/opus
$P ch17/ws_transport.py run --codec opus --jitter wifi --out $BOOK_DATA/runs/ch17_ws/opus_wifi
$P ch17/ws_transport.py run --codec opus --jitter mobile --out $BOOK_DATA/runs/ch17_ws/opus_mobile

# ch18 (one GPU with 24 GB, about 75 minutes; three servers: TTS 8091, LLM 8000, ASR 6010)
# -> ch18_cascade_<name>.json for each --name below
ch18/run_cascade.sh <logdir> \
  "$P ch18/cascade.py --name cascade_smoke --turns 4" \
  "$P ch18/cascade.py --name cascade_sil500 --endpoint silence --silence-ms 500 --save-audio" \
  "$P ch18/cascade.py --name cascade_sil300 --endpoint silence --silence-ms 300" \
  "$P ch18/cascade.py --name cascade_sil800 --endpoint silence --silence-ms 800" \
  "$P ch18/cascade.py --name cascade_smart --endpoint smart --trigger-ms 200 --fallback-ms 3000" \
  "$P ch18/cascade.py --name cascade_bargein --barge-in --turns 20 --endpoint silence --silence-ms 500"
# a second server start: sentence prefetch, then the 500 ms setting again on the same servers
ch18/run_cascade.sh <logdir> \
  "$P ch18/cascade.py --name cascade_sil500_prefetch --endpoint silence --silence-ms 500 --tts-prefetch" \
  "$P ch18/cascade.py --name cascade_sil500_v2 --endpoint silence --silence-ms 500"
```

A dry run of the cascade logic on the CPU, with no GPU: stand-ins for the LLM (port 18000) and TTS (port 18091)
servers, the chapter 15 sherpa-onnx server as the ASR (port 6006), then four turns. Its summary goes to
`$BOOK_DATA/runs/cascade_dryrun/`, not to `results/`; no number from it is reported. Three terminals, each in `code/serve` with the ssh-serve environment and `M`, `MODEL` set as above:

```bash
python ch18/mock_servers.py 18000 18091
python ch15/sherpa_server.py --port 6006 --use-endpoint 0 --max-batch-size 3 --num-threads 2 --nn-pool-size 1 $MODEL
python ch18/cascade.py --name cascade_dryrun --turns 4 --no-lang-hint --results-dir $BOOK_DATA/runs/cascade_dryrun \
    --asr-url ws://127.0.0.1:6006 --llm-url http://127.0.0.1:18000 --tts-url http://127.0.0.1:18091
```
