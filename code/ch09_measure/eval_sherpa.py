"""Chapter 9 (CPU track): accuracy, stability and latency of a sherpa-onnx streaming Zipformer on a
fixed subset.

    python code/ch09_measure/eval_sherpa.py --manifest librispeech-test-clean-200 \
        [--decoding greedy_search|modified_beam_search] [--hotwords FILE] [--int8] [--name NAME]
        [--tail-padding-s 0.66] [--runs-suffix SUFFIX] [--results-dir DIR]

Streams each utterance in 160 ms chunks with simulated real-time pacing (hbasr.latency), one
recognizer thread, and writes
  $BOOK_DATA/runs/<name><suffix>/utts.jsonl   every chunk of every utterance
  <results-dir>/<name>.json                                 the summary below
Summary: WER (English, words) or CER (Mandarin, characters) under each normalizer in
hbasr.textnorm; UPWR/UPSR; first-token delay, final latency and RTF percentiles; emission lag
relative to the model's own token timestamps.
"""
import argparse
import os

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import evaluate, provenance, textnorm  # noqa: E402
from hbasr.backends_sherpa import TAIL_PADDING_S, SherpaConfig, SherpaRecognizer, char_times, word_start_times  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--decoding", default="greedy_search")
ap.add_argument("--max-active-paths", type=int, default=4)
ap.add_argument("--hotwords", default="")
ap.add_argument("--hotwords-score", type=float, default=1.5)
ap.add_argument("--int8", action="store_true")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--chunk-ms", type=float, default=160.0)
ap.add_argument("--limit", type=int)
ap.add_argument("--name")
ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
ap.add_argument("--tail-padding-s", type=float, default=TAIL_PADDING_S,
                help="synthetic silence appended before input_finished() (default: sherpa-onnx's own 0.66 s)")
ap.add_argument("--runs-suffix", default="",
                help="appended to the raw-output directory name under $BOOK_DATA/runs/, to keep a variant apart")
args = ap.parse_args()

lang = evaluate.lang_of(args.manifest)
cfg = SherpaConfig(model="zipformer-zh" if lang == "zh" else "zipformer-en", int8=args.int8,
                   decoding_method=args.decoding, max_active_paths=args.max_active_paths,
                   num_threads=args.threads, hotwords_file=args.hotwords, hotwords_score=args.hotwords_score,
                   tail_padding_s=args.tail_padding_s)
name = args.name or "-".join(["sherpa", cfg.model, args.manifest, args.decoding] + (["int8"] if args.int8 else [])
                              + (["hotwords"] if args.hotwords else []))
rec = SherpaRecognizer(cfg)


def unit_times(r):
    toks, times = r.tokens_and_times()
    return word_start_times(toks, times) if lang == "en" else char_times(
        toks, times, lambda t: textnorm.zh_units(textnorm.ZH_DEFAULT(t)))


prov = provenance.collect({"manifest": args.manifest, "recognizer": cfg.to_dict(), "chunk_ms": args.chunk_ms,
                           "pace": "simulated", "track": "CPU"})
path = evaluate.stream_corpus(rec, args.manifest, name + args.runs_suffix, args.chunk_ms, args.limit,
                              unit_times=unit_times)
res = evaluate.score(path, lang, args.manifest)
res["provenance"] = prov | {"loadavg_at_end": os.getloadavg()}
provenance.write_json(os.path.join(args.results_dir, name + ".json"), res)
a = res["accuracy"]
print(name, {k: round(v["rate"] * 100, 2) for k, v in a.items()},
      "UPWR", round(res["stability"]["UPWR"], 3), "UPSR", round(res["stability"]["UPSR"], 3),
      "first-token p50", round(res["latency"]["first_token_delay_s"]["p50"], 3),
      "final p50", round(res["latency"]["final_latency_after_speech_end_s"]["p50"], 3),
      "RTF", round(res["latency"]["corpus_rtf"], 4))
