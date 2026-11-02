"""Chapters 5 and 9 (GPU track): WER, stability and latency of a NeMo cache-aware streaming model at
several att_context_size settings (one RTX 4090).

    python code/ch09_measure/eval_nemo.py --model fastconformer-en-multi \
      --manifests librispeech-test-clean-200 librispeech-test-other-200 --decoders rnnt ctc

For every (manifest, decoder, att_context_size) the corpus is streamed in 160 ms chunks with simulated
real-time pacing through hbasr.backends_nemo (incremental features, an encoder step as soon as its
frames exist, CUDA synchronized after every chunk so compute time is real). Raw logs go to
$BOOK_DATA/runs/<name>/utts.jsonl, summaries to results/<name>.json, and a
run-time sweep table to results/<model>-sweep.json (make_tables.py rebuilds the table from the
per-run summaries after any rescoring; that table, summary_table.json, is the one to cite).
"""
import argparse
import json
import os

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import evaluate, provenance  # noqa: E402
from hbasr.backends_nemo import NemoCacheAwareRecognizer, NemoConfig, load_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="fastconformer-en-multi")
ap.add_argument("--manifests", nargs="+", default=["librispeech-test-clean-200", "librispeech-test-other-200"])
ap.add_argument("--decoders", nargs="+", default=["rnnt", "ctc"])
ap.add_argument("--contexts", nargs="*", help="e.g. 70,13 70,0 ; default: every context the model supports")
ap.add_argument("--target-lang", default="", help="prompt for Nemotron 3.5, e.g. zh-CN")
ap.add_argument("--limit", type=int)
ap.add_argument("--chunk-ms", type=float, default=160.0)
ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
args = ap.parse_args()

model = load_model(args.model, "cuda")
ctxs = ([tuple(int(v) for v in c.split(",")) for c in args.contexts] if args.contexts
        else [tuple(c) for c in model.encoder.att_context_size_all])
table = []
for manifest in args.manifests:
    lang = evaluate.lang_of(manifest)
    for dec in args.decoders:
        for ctx in ctxs:
            cfg = NemoConfig(model=args.model, att_context_size=ctx, decoder=dec, target_lang=args.target_lang)
            rec = NemoCacheAwareRecognizer(cfg, model=model)
            # warm-up so CUDA kernel selection is not charged to the first utterance
            import numpy as np
            rec.start(); rec.accept(np.zeros(32000, np.float32)); rec.finish()
            name = f"nemo-{args.model}-{manifest}-{dec}-ctx{ctx[0]}_{ctx[1]}"
            prov = provenance.collect({"manifest": manifest, "recognizer": cfg.to_dict(),
                                       "streaming_cfg": rec.describe(), "chunk_ms": args.chunk_ms,
                                       "pace": "simulated", "track": "GPU",
                                       "gpu_torch": torch.cuda.get_device_name(0)})
            path = evaluate.stream_corpus(rec, manifest, name, args.chunk_ms, args.limit)
            res = evaluate.score(path, lang, manifest)
            res["provenance"] = prov | {"loadavg_at_end": os.getloadavg()}
            provenance.write_json(os.path.join(args.results_dir, name + ".json"), res)
            key = "lower+punct" if lang == "en" else "full"
            row = {"manifest": manifest, "decoder": dec, "att_context_size": list(ctx),
                   "streaming_cfg": rec.describe(),
                   "error_rate": res["accuracy"][key]["rate"], "normalizer": key,
                   "UPWR": res["stability"]["UPWR"], "UPSR": res["stability"]["UPSR"],
                   "first_token_delay_p50_s": res["latency"]["first_token_delay_s"]["p50"],
                   "first_token_delay_p90_s": res["latency"]["first_token_delay_s"]["p90"],
                   "final_after_speech_end_p50_s": res["latency"]["final_latency_after_speech_end_s"]["p50"],
                   "final_after_speech_end_p90_s": res["latency"]["final_latency_after_speech_end_s"]["p90"],
                   "final_after_input_end_p50_s": res["latency"]["final_latency_after_input_end_s"]["p50"],
                   "corpus_rtf": res["latency"]["corpus_rtf"], "summary": name + ".json"}
            table.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("streaming_cfg",)}), flush=True)
provenance.write_json(os.path.join(args.results_dir, f"{args.model}-sweep.json"),
                      {"provenance": provenance.collect({"gpu_torch": torch.cuda.get_device_name(0)}), "rows": table})
