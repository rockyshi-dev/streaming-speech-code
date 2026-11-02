# Hands-on code

| Track | Directory | Chapters |
|---|---|---|
| speech recognition (this file) | `hbasr/`, `ch01_audio/` ... `ch09_measure/`, `data/`, `env/`, `models/`, `tests/`, `tools/` | 1-7, 9 |
| Whisper / LLM-based streaming ASR | `ch08_whisper/la2_trace.py` | 8 |
| speech synthesis | `tts/` (its own README) | 10-14 |
| serving, transport, cascade | `serve/` (its own README) | 15-18 |

## Speech recognition, chapters 1-9

Everything the ASR chapters ask the reader to run, plus the harness that produced the handbook's own
measurements. CPU track: sherpa-onnx with icefall streaming Zipformer transducers. GPU track: NVIDIA
NeMo cache-aware FastConformer and Nemotron 3.5 ASR on one RTX 4090.

## Layout

| Path | What | Env |
|---|---|---|
| `hbasr/` | the shared library (importable as `hbasr` after `pip install -e code`) | both |
| `hbasr/audio.py` | seconds/samples/bytes, PCM byte layout, resampling, 160 ms chunking | asr |
| `hbasr/features.py` | numpy Kaldi-compatible log-mel; utterance vs causal vs sliding normalization | asr |
| `hbasr/alignment.py` | CTC and transducer forward algorithm, brute-force alignment enumeration, CTC greedy | asr |
| `hbasr/textnorm.py` | the stated normalizers (`en-raw`, `en-lower`, `en-lower+punct`; `zh-raw` ... `zh-full`) | both |
| `hbasr/metrics.py` | edit distance, WER/CER, UPWR/UPSR, percentiles | both |
| `hbasr/latency.py` | the latency harness (simulated or real-time pacing); reusable by chapters 15 and 18 | both |
| `hbasr/evaluate.py` | stream a manifest through a recognizer, write raw logs, score them | both |
| `hbasr/backends_sherpa.py` | sherpa-onnx adapter | asr |
| `hbasr/backends_nemo.py` | incremental NeMo cache-aware adapter (features and encoder steps only when their audio exists) | nemo |
| `hbasr/vad.py` | Silero VAD (offline segments, streaming iterator) | asr |
| `data/` | `make_manifests.py` and the committed fixed subsets `manifests/*.tsv` | asr |
| `models/fetch_models.sh` | fetch the pre-trained models at pinned revisions into `$BOOK_MODELS` | - |
| `env/` | lock files (`*.requirements.lock.txt`, `*.conda.explicit.txt`) and `create_envs.sh` | - |
| `ch01_audio/` | `audio_numbers.py`, `speech_pitch.py` (pitch period and f0 by autocorrelation) | asr |
| `ch02_features/` | `frontend_compare.py`, `normalization_leak.py`, `model_frontends.py`, `frontend_mismatch_nemo.py` (ssh-nemo env, CPU) | asr / nemo |
| `ch03_latency/` | `harness_demo.py` (harness on one utterance; att_context_size to ms table) | asr |
| `ch04_alignment/` | `ctc_rnnt_by_hand.py`, `chapter_examples.py`; `export_posteriors.py` (ssh-nemo env, CPU) then `ctc_greedy_by_hand.py` | asr / nemo |
| `ch05_streaming_gpu/` | `check_adapter.py`: the incremental NeMo adapter against NeMo's own streaming loop | nemo, GPU |
| `ch06_decoding/` | `partials_finals.py` (partial waterfall), `hotwords.py` (contextual biasing) | asr |
| `ch07_endpoint/` | `endpoint_sweep.py` (sherpa-onnx rule 2 vs Silero VAD trailing silence), `endpoint_vs_alignment.py`, `silero_packet_rounding.py`; `run_ch06_ch07.sh` | asr |
| `ch08_whisper/` | `la2_trace.py` (LocalAgreement-2 traced by hand, no dependencies) | asr |
| `ch09_measure/` | `eval_sherpa.py`, `eval_nemo.py`, `rescore.py`, `breakdowns.py`, `make_tables.py`, `run_cpu_all.sh`, `run_gpu_all.sh` | asr / nemo |
| `tests/` | pytest, CPU only | asr |
| `tools/roots.sh` | sourced by the shell scripts: the three directories below, and how conda is found | - |

Each chapter directory has a `results/` folder with small summary JSON files (committed). Raw,
per-utterance output goes under `$BOOK_DATA/runs/<name>/` (not committed). Every
summary carries a `provenance` block: command, package versions, CPU/GPU, load average, git commit. Paths in
result files are written as `$BOOK_DATA/...`, `$BOOK_DATASETS/...`, `$BOOK_MODELS/...` or relative to the
repository root.

## Setting up

The code finds everything through three directories, each set by an environment variable:

| Variable | Holds | Default |
|---|---|---|
| `BOOK_DATA` | run output (`runs/`), prepared TTS inputs (`inputs/`), fetched vendor files (`vendor/`), derived files and caches | `work/` at the repository root |
| `BOOK_DATASETS` | the corpora, read in place | `$BOOK_DATA/datasets` |
| `BOOK_MODELS` | pre-trained model directories | `$BOOK_DATA/models` |

With the defaults everything lives in `work/` (ignored by git): download the corpora into
`work/datasets/` in the layout below and the models into `work/models/`. To keep them elsewhere, point the
variables there instead, for example:

```bash
export BOOK_DATA=$HOME/book-data BOOK_DATASETS=$HOME/corpora BOOK_MODELS=$HOME/models
```

The scripts under `code/` apply the defaults themselves (`code/tools/roots.sh`). The commands you type from
these READMEs use the variables, so export them in your shell; for the defaults, from the repository root:

```bash
export BOOK_DATA=$PWD/work BOOK_DATASETS=$PWD/work/datasets BOOK_MODELS=$PWD/work/models
```

The scripts find conda through `CONDA_BASE` (the conda installation directory); if that is unset, through
`conda info --base`, or else in `~/miniconda3`, `~/miniforge3`, `~/anaconda3` or `~/mambaforge`.

Caches go under `$BOOK_DATA/cache/`. Models hosted on the Hugging Face Hub (Nemotron 3.5 ASR, parakeet-ctc,
Qwen3-TTS, Kokoro, the codecs) are read from the Hugging Face cache at `$HF_HOME`, which defaults to
`$BOOK_DATA/cache/huggingface`. `code/models/fetch_models.sh` puts the sherpa-onnx and FastConformer models into
`$BOOK_MODELS` at pinned revisions (about 1.1 GB); `code/models/fetch_models.sh --serve` adds the models of
chapters 15-18 (about 4 GB more). The two ASR models read from the Hugging Face cache are fetched at the
revisions the code expects with the `hf` command of `huggingface_hub` (env `ssh-nemo`):

```bash
export HF_HOME=${HF_HOME:-$BOOK_DATA/cache/huggingface}
hf download nvidia/nemotron-3.5-asr-streaming-0.6b --revision ea30d66debe3740a08b573244286791d423d6b3e
hf download nvidia/parakeet-ctc-0.6b --revision ad09ba1cc62743fbc9814de5d2016fca9096485a
```

Corpora, and the layout the committed manifests expect under `$BOOK_DATASETS` (manifest paths are relative
to it):

| Corpus | Source | Layout under `$BOOK_DATASETS` |
|---|---|---|
| LibriSpeech test-clean, test-other | openslr.org/12 (`test-clean.tar.gz`, `test-other.tar.gz`) | `librispeech/extracted/LibriSpeech/test-clean/...`, `.../test-other/...` (the tarballs unpacked into `librispeech/extracted/`) |
| AISHELL-1 test | openslr.org/33 (`data_aishell.tgz`) | `aishell1/transcript/aishell_transcript_v0.8.txt` and `aishell1/wav/test/<speaker>/*.wav` (unpack `data_aishell.tgz`, rename `data_aishell/` to `aishell1/`, and unpack the per-speaker archives in `aishell1/wav/`) |
| LibriTTS-R test-clean (chapters 11-14) | openslr.org/141 (`test_clean.tar.gz`) | `libritts-r/test_clean.tar.gz`, read without unpacking |

The chapter 11-14 reference clips are copied from these corpora into `$BOOK_DATA/inputs/` by
`tts/textset/build_textset.py`; chapter 15 fetches its server files with `serve/ch15/fetch_vendor.sh`;
`serve/README.md` lists the models and files chapters 15-18 read from `$BOOK_MODELS` and `$BOOK_DATA/vendor/`.

```bash
bash code/env/create_envs.sh           # ssh-asr (CPU) and ssh-nemo (GPU), conda-forge + pip locks
bash code/models/fetch_models.sh       # ~1.1 GB into $BOOK_MODELS (Nemotron 3.5 is read from the HF cache)
conda activate ssh-asr
python code/data/make_manifests.py     # only to regenerate; the manifests are committed
python -m pytest -q code/tests
```

## Running

Run these from the repository root, in this order. Each step writes the result files the chapters cite.

1. Single-utterance examples, CPU (env `ssh-asr`):

   ```bash
   python code/ch01_audio/audio_numbers.py
   python code/ch01_audio/speech_pitch.py
   python code/ch02_features/frontend_compare.py
   python code/ch02_features/normalization_leak.py
   python code/ch02_features/model_frontends.py
   python code/ch03_latency/harness_demo.py
   python code/ch04_alignment/ctc_rnnt_by_hand.py
   python code/ch04_alignment/chapter_examples.py
   python code/ch06_decoding/partials_finals.py
   python code/ch08_whisper/la2_trace.py
   ```

2. NeMo on the CPU (env `ssh-nemo`; `CUDA_VISIBLE_DEVICES=` keeps the GPU out of it), then the chapter 4
   greedy decode by hand (env `ssh-asr`):

   ```bash
   CUDA_VISIBLE_DEVICES= python code/ch02_features/frontend_mismatch_nemo.py --n 50
   CUDA_VISIBLE_DEVICES= python code/ch04_alignment/export_posteriors.py
   conda activate ssh-asr && python code/ch04_alignment/ctc_greedy_by_hand.py
   ```

3. Reference word times by forced alignment (env `ssh-asr`, CPU). The word latency of chapter 9 and the English
   endpoint table of chapter 7 are measured against them; without this step those numbers are left out:

   ```bash
   PYTHONPATH=code python -m hbasr.align_ref librispeech-test-clean-200 librispeech-test-other-200
   ```

4. The corpus runs. CPU track (env `ssh-asr`): greedy and beam search on the three subsets (about 20 minutes),
   then the endpoint sweep and the hotword runs (about 30 minutes):

   ```bash
   code/ch09_measure/run_cpu_all.sh
   code/ch07_endpoint/run_ch06_ch07.sh
   ```

   The chapter 6 tail-padding variant: the same CPU runs with 1.5 s of synthetic tail silence instead of
   sherpa-onnx's 0.66 s, raw output in separate `$BOOK_DATA/runs/<name>-tail1.5/` directories, summaries in
   `code/ch09_measure/results/tail_padding_1.5s/`. The two hotword runs read the 1.5 s beam-search baselines,
   so they come last:

   ```bash
   T="--tail-padding-s 1.5 --runs-suffix=-tail1.5 --results-dir code/ch09_measure/results/tail_padding_1.5s"
   python code/ch09_measure/eval_sherpa.py --manifest librispeech-test-clean-200 --decoding greedy_search $T
   python code/ch09_measure/eval_sherpa.py --manifest librispeech-test-clean-200 --decoding modified_beam_search $T
   python code/ch09_measure/eval_sherpa.py --manifest librispeech-test-other-200 --decoding greedy_search $T
   python code/ch09_measure/eval_sherpa.py --manifest librispeech-test-other-200 --decoding modified_beam_search $T
   python code/ch09_measure/eval_sherpa.py --manifest aishell1-test-200 --decoding greedy_search $T
   python code/ch09_measure/eval_sherpa.py --manifest aishell1-test-200 --decoding modified_beam_search $T
   python code/ch06_decoding/hotwords.py --manifest librispeech-test-other-200 $T
   python code/ch06_decoding/hotwords.py --manifest aishell1-test-200 $T
   ```

   GPU track (env `ssh-nemo`, one GPU): the adapter check (about 10 minutes), then every GPU run of chapters 5
   and 9 (about an hour on an RTX 4090). Run the CPU steps 1-4 above first: the GPU runs read the Silero speech
   regions that the first CPU corpus run caches in `$BOOK_DATA/runs/vad/`, and `ssh-nemo` does not contain
   `silero-vad`, so without that cache they stop with `No module named 'silero_vad'`:

   ```bash
   python code/ch05_streaming_gpu/check_adapter.py
   code/ch09_measure/run_gpu_all.sh
   ```

5. What is computed from the raw logs (env `ssh-asr`, CPU, minutes): re-score every summary with the forced
   alignment, the chapter 9 breakdowns, the chapter 7 endpoint readings against the alignment, and the summary
   table:

   ```bash
   python code/ch09_measure/rescore.py
   python code/ch09_measure/breakdowns.py
   python code/ch07_endpoint/endpoint_vs_alignment.py
   python code/ch07_endpoint/silero_packet_rounding.py
   python code/ch09_measure/make_tables.py
   ```

## What the latency numbers are

`hbasr/latency.py` defines them; in short, for one utterance streamed in 160 ms chunks:

- clock: t = 0 at the first captured sample; a chunk covering [a, b) arrives at t = b;
- simulated pacing: start = max(arrival, previous done), done = start + measured compute time (the
  schedule an otherwise idle machine would produce live; `harness_demo.py` checks it against a
  real-time run);
- first-token delay = time of the first non-empty partial result - speech onset (offline Silero VAD);
- final latency after end of speech = time of the final result - last speech sample (Silero VAD);
  final latency after input end = the same minus the end of the audio the client sent;
- endpoint delay = time the endpointer fired - end of speech;
- RTF = total compute time / audio duration (corpus RTF = sum of compute / sum of audio);
- emission lag = time a word (Mandarin: character) became stable in the partial results - the time
  the model itself assigns to it (its token timestamp); a delay inside the recognizer, not against a
  forced alignment.

Not included: network, client buffering, model load. CPU numbers use one recognizer thread; each summary
records the load average at start and end.

## Models and licences

| Model | Source (pinned revision) | Licence | Used for |
|---|---|---|---|
| sherpa-onnx streaming Zipformer, English (icefall, LibriSpeech 960 h) | `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26@672fbf1b` | Apache-2.0 | CPU track, English |
| sherpa-onnx streaming Zipformer, Mandarin (icefall multi_zh-hans) | `k2-fsa/sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12@ac54a23c` | Apache-2.0 (weights; on the training data see appendix C) | CPU track, Mandarin |
| NeMo FastConformer hybrid large streaming multi (cache-aware, transducer + CTC) | `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi@ae981433` | CC-BY-4.0 | GPU track, English; ch02, ch04 on CPU |
| Nemotron 3.5 ASR streaming 0.6B | `nvidia/nemotron-3.5-asr-streaming-0.6b@ea30d66d` (HF cache) | OpenMDW-1.1 | GPU track, Mandarin |
| Silero VAD 6.2.3 (ONNX file inside the `silero-vad` wheel) | PyPI `silero-vad==6.2.3` | MIT (wheel licence file) | ch07, speech onset/offset |
| parakeet-ctc-0.6b | config only, read from the HF cache | CC-BY-4.0 | ch02: shows an offline model's per-utterance normalization |

Corpora (read in place): LibriSpeech test-clean/test-other (CC BY 4.0, `$BOOK_DATASETS/librispeech`,
tarball md5s match OpenSLR's `md5sum.txt`), AISHELL-1 test (Apache-2.0,
`$BOOK_DATASETS/aishell1`, OpenSLR 33). Text tools: cn2an (MIT),
opencc-python-reimplemented (Apache-2.0), kaldi-native-fbank (Apache-2.0), jiwer (Apache-2.0).

## Serving, transport and the voice cascade (chapters 15-18)

`serve/` holds the hands-on code for chapters 15-18: streaming ASR servers (sherpa-onnx on CPU, a batched
NeMo cache-aware server on the GPU) and a load generator, TTS load tests, Opus / jitter-buffer / WebSocket
transport, and the timed voice cascade with barge-in. See `serve/README.md` (layout, environments, models,
licences, and the exact commands under "How to run"); the measured numbers are in `serve/results/` and
chapters 15-18. Tests:
`conda run -n ssh-serve python -m pytest -q code/serve/tests`.
