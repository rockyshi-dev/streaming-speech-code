# Hands-on code, chapters 10-14 (streaming text-to-speech)

Everything the TTS chapters ask the reader to run, plus the harness that produced the handbook's
own TTS measurements. CPU track: Kokoro-82M (English v1.0, Mandarin v1.1-zh). GPU track: vLLM-Omni
0.32.0 serving Qwen3-TTS-12Hz-1.7B-CustomVoice on one RTX 4090.

## Layout

| Path | What | Env |
|---|---|---|
| `hbtts/audio.py` | first audible sample (first 10 ms window at or above -40 dBFS), WAV header stripping for streamed bytes | any |
| `hbtts/stream.py` | per-request client record: TTFA, first PCM byte, RTF, inter-arrival, real-time player simulation (underruns) | any |
| `hbtts/textstream.py` | sentence splitting, text units (word / CJK character), fixed-rate release (simulated LLM tokens) | any |
| `hbtts/errorrate.py` | edit operations, WER / CER, normalization, pooled rates | any |
| `hbtts/runinfo.py` | provenance block in every summary JSON (command, versions, git commit, CPU/GPU, load average) | any |
| `textset/` | `build_textset.py` and the committed fixed text set `en.tsv`, `zh.tsv`, `prompts.tsv` | cpu |
| `ch11_codecs/codec_roundtrip.py` | Mimi / EnCodec / DAC: frame rate, codebooks, bit/s, reconstruction, CPU RTF; mel and token frame rates | cpu |
| `ch12_kokoro_cpu/kokoro_stream.py` | Kokoro sentence-level streaming: TTFA vs sentence length, paragraphs whole / per sentence / text streamed in | cpu |
| `ch13_chunk_boundaries/chunk_boundaries.py` | Vocos chunked with/without context and crossfade; Mimi stateless vs streaming decode; error at the joins | cpu |
| `gpu/with_server.sh` | start vLLM-Omni, wait for `/health`, run one client command, stop the server | gpu |
| `gpu/qwen3tts_client.py` | client: `smoke`, `input-modes` (chapter 12), `concurrency` (chapter 14) | gpu |
| `gpu/run_all_gpu.sh` | one server start, then the chapter 12 input-mode run | gpu |
| `gpu/run_concurrency_gpu.sh` | the chapter-14 sweep (N = 1..64) with its own server | gpu |
| `gpu/run_concurrency_then_eval.sh` | the sweep, then the intelligibility evaluation, in one GPU job | gpu |
| `ch14_harness/run_eval_gpu.sh` | intelligibility of every set on the GPU | eval |
| `ch14_harness/eval_intelligibility.py` | Whisper-large-v3 WER (en), Paraformer-zh CER (zh) for synthesized and original speech | eval |
| `envs/` | `create_envs.sh`, `*.lock.txt` (pip freeze), `*.conda.txt` (conda explicit) | - |
| `tests/` | pytest, CPU only, no models: the harness logic | any |
| `results/` | committed summary JSON; `ch13_audio/` holds 3-second listening clips | - |

Raw per-request output (rows.jsonl, every wav) goes to `$BOOK_DATA/runs/<name>/`.
Model weights come from the HF cache (`HF_HOME`); the GPU wrapper points every vLLM, Triton,
inductor and flashinfer cache at `$BOOK_DATA/cache/`.

## Run

Run from the repository root. The GPU steps need one GPU with 24 GB.

The models come from the Hugging Face cache (`$HF_HOME`, default `$BOOK_DATA/cache/huggingface`) at the
revisions the code loads (`hbtts/hub.py`). Fetch them once, after `create_envs.sh` (below) has built the
environments, in any of the three (each has the `hf` command of `huggingface_hub`). Of moshiko only the Mimi
file is used:

```bash
export HF_HOME=${HF_HOME:-$BOOK_DATA/cache/huggingface}
hf download kyutai/mimi --revision 89091b3e466eb6a9d11e537bf26b144f194978f7
hf download facebook/encodec_24khz --revision c1dbe2ae3f1de713481a3b3e7c47f357092ee040
hf download descript/dac_24khz --revision 6ba020b5ba7d9d8076fb90db7e67f27e31980f6e
hf download charactr/vocos-mel-24khz --revision 0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21
hf download kyutai/moshiko-pytorch-bf16 tokenizer-e351c8d8-checkpoint125.safetensors --revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289
hf download hexgrad/Kokoro-82M --revision f3ff3571791e39611d31c381e3a41a3af07b4987
hf download hexgrad/Kokoro-82M-v1.1-zh --revision 01e7505bd6a7a2ac4975463114c3a7650a9f7218
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --revision 0c0e3051f131929182e2c023b9537f8b1c68adfe
hf download openai/whisper-large-v3 --revision 06f233fe06e710322aca913c1bc4249a0d71fce1
hf download funasr/paraformer-zh --revision d7811ee3ac581fbcfdeb37c98c6ba674028433dc
```

```bash
bash code/tts/envs/create_envs.sh                   # ssh-tts-cpu, ssh-tts-gpu, ssh-tts-eval
conda activate ssh-tts-cpu
cd code/tts
python -m pytest -q tests
python textset/build_textset.py                      # copies the reference clips into $BOOK_DATA/inputs/ (the text set is committed)
python ch11_codecs/codec_roundtrip.py --threads 4    # -> results/ch11_codecs.json
python ch12_kokoro_cpu/kokoro_stream.py --threads 4  # -> results/ch12_kokoro_cpu.json
python ch13_chunk_boundaries/chunk_boundaries.py     # -> results/ch13_chunk_boundaries.json, results/ch13_audio/
conda activate ssh-tts-eval                          # intelligibility of the Kokoro set on the CPU
python ch14_harness/eval_intelligibility.py --cpu --threads 4 --name ch14_intelligibility_cpu \
    --source human --source kokoro:$BOOK_DATA/runs/ch12_kokoro_cpu/rows.jsonl    # -> results/ch14_intelligibility_cpu.json
cd ../..
code/tts/gpu/run_all_gpu.sh
    # chapter 12 input modes -> results/ch12_qwen3tts_input_modes.json
code/tts/gpu/run_concurrency_gpu.sh
    # chapter 14 sweep, N = 1..64, per-level warm-up burst (about 15 minutes) -> results/ch14_qwen3tts_concurrency.json
code/tts/ch14_harness/run_eval_gpu.sh
    # the intelligibility evaluation (about 25 minutes): Whisper-large-v3 / Paraformer-zh on every
    # synthesized set and the original recordings -> results/ch14_intelligibility.json
    # (code/tts/gpu/run_concurrency_then_eval.sh runs these two in one go)
```

Definitions (TTFA, RTF, underrun) are in the docstring of `hbtts/stream.py`; the measured numbers are in
`results/` and in chapters 11-14.
