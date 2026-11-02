"""Chapter 7: how the endpoint delay and the false cut-offs move with the trailing-silence setting,
for sherpa-onnx's rule-based endpointer and for Silero VAD, on the same audio.

    python code/ch07_endpoint/endpoint_sweep.py [--n 100]

Material: the first N utterances of librispeech-test-clean-200 and of aishell1-test-200, each with
2.0 s of near-silence appended (Gaussian noise at about -80 dBFS) so an endpoint can happen after
the last word. "End of speech" is the last speech sample found by offline Silero VAD on the
original utterance (threshold 0.5, 100 ms minimum silence).

sherpa-onnx endpointing (rule-based, on decoded blanks): rule 2 fires when at least one token has
been decoded and the decoder has emitted only blanks for `rule2_min_trailing_silence` seconds.
Rule 1 (2.4 s, nothing decoded) and rule 3 (20 s utterance) stay at their defaults. Swept: rule 2 in
{0.2, 0.4, 0.6, 0.8, 1.0, 1.2} s. Audio is streamed in 160 ms chunks with simulated real-time pacing.

Silero VAD (streaming VADIterator on 32 ms windows): speech ends after `min_silence_duration_ms`
below threshold - 0.15. Swept: {100, 200, 300, 500, 800, 1000} ms.

For each setting and language:
  endpoint delay  = time the first endpoint decision is available - end of speech (p50/p90/mean),
                    over utterances where the first endpoint came after the end of speech;
  false cut-offs  = utterances whose first endpoint came more than 50 ms before the end of speech
                    (a pause inside the sentence was taken for its end).
Writes results/endpoint_sweep.json and per-utterance rows to runs/ch07/endpoint_sweep.jsonl.
"""
import argparse
import json
import os

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, data, evaluate, latency, metrics, paths, provenance, vad  # noqa: E402
from hbasr.backends_sherpa import SherpaConfig, SherpaRecognizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{paths.BOOK_DATA}/runs/ch07"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)

RULE2 = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
VAD_MS = [100, 200, 300, 500, 800, 1000]
TOL = 0.05
TAIL_S = 2.0
rng = np.random.default_rng(0)

rows = []
summary = {"provenance": provenance.collect({"n_per_language": args.n, "tail_s": TAIL_S}), "sherpa": {}, "silero": {},
           "material": {}}
for manifest, model in [("librispeech-test-clean-200", "zipformer-en"), ("aishell1-test-200", "zipformer-zh")]:
    lang = evaluate.lang_of(manifest)
    utts = data.read_manifest(manifest)[: args.n]
    regions = evaluate.vad_regions(manifest, data.read_manifest(manifest))
    clips = []
    gaps = []
    for u in utts:
        x, sr = audio.read_wav(u.path)
        tail = (1e-4 * rng.standard_normal(int(TAIL_S * sr))).astype(np.float32)
        clips.append((u, np.concatenate([x, tail]), regions[u.utt_id][1]))
        segs = vad.speech_segments(x, sr)
        gaps += [b[0] - a[1] for a, b in zip(segs, segs[1:])]
    summary["material"][lang] = {"utterances": len(clips),
                                 "internal_pauses_ge_200ms": int(sum(g >= 0.2 for g in gaps)),
                                 "internal_pauses_ge_500ms": int(sum(g >= 0.5 for g in gaps)),
                                 "internal_pauses_ge_1000ms": int(sum(g >= 1.0 for g in gaps)),
                                 "longest_internal_pause_s": float(max(gaps) if gaps else 0.0)}

    # sherpa-onnx rule 2 sweep
    summary["sherpa"][lang] = {}
    for r2 in RULE2:
        rec = SherpaRecognizer(SherpaConfig(model=model, enable_endpoint=True, rule2_min_trailing_silence=r2))
        delays, cut, none = [], 0, 0
        for u, x, off in clips:
            lg = latency.run_stream(rec, x, u.utt_id, 160.0, "simulated", stop_at_endpoint=True)
            ep = lg.endpoint_s
            row = {"lang": lang, "system": "sherpa-rule2", "setting": r2, "utt_id": u.utt_id, "speech_end_s": off,
                   "endpoint_s": ep, "text_at_endpoint": lg.endpoint_text}
            if ep is None:
                none += 1
            elif ep < off - TOL:
                cut += 1
                row["false_cutoff"] = True
            else:
                delays.append(ep - off)
            rows.append(row)
        summary["sherpa"][lang][str(r2)] = {"endpoint_delay_s": metrics.percentiles(delays),
                                            "false_cutoffs": cut, "no_endpoint": none, "utterances": len(clips)}
        print(lang, "sherpa rule2", r2, json.dumps(summary["sherpa"][lang][str(r2)]), flush=True)

    # Silero VAD sweep
    summary["silero"][lang] = {}
    for ms in VAD_MS:
        sv = vad.StreamingVAD(16000, 0.5, ms)
        delays, cut, none = [], 0, 0
        for u, x, off in clips:
            ends = [e for e in sv.run(x) if e["kind"] == "end"]
            first = ends[0]["decided_at_s"] if ends else None
            row = {"lang": lang, "system": "silero", "setting": ms, "utt_id": u.utt_id, "speech_end_s": off,
                   "endpoint_s": first}
            if first is None:
                none += 1
            elif first < off - TOL:
                cut += 1
                row["false_cutoff"] = True
            else:
                delays.append(first - off)
            rows.append(row)
        summary["silero"][lang][str(ms)] = {"endpoint_delay_s": metrics.percentiles(delays),
                                            "false_cutoffs": cut, "no_endpoint": none, "utterances": len(clips)}
        print(lang, "silero", ms, json.dumps(summary["silero"][lang][str(ms)]), flush=True)

with open(os.path.join(OUT, "endpoint_sweep.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
summary["raw"] = os.path.join(OUT, "endpoint_sweep.jsonl")
provenance.write_json(os.path.join(HERE, "results", "endpoint_sweep.json"), summary)
