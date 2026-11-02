"""Chapter 3: the latency harness on one utterance, and the look-ahead arithmetic table.

    python code/ch03_latency/harness_demo.py [--model zipformer-en] [--chunk-ms 160]

1. Streams one LibriSpeech utterance into the sherpa-onnx streaming Zipformer (CPU, 1 thread),
   once at real-time pace (sleeping until each chunk's audio exists) and once with simulated pacing
   (no sleeping, virtual clock), and prints the per-chunk log: audio time, when the chunk was
   processed, the partial result after it. Then the per-utterance numbers: first-token delay, final
   latency, RTF. The two pacings should agree to within scheduling noise.
2. Prints the att_context_size -> milliseconds table for 4x and 8x subsampling (10 ms feature hop).
Writes results/harness_demo.json and the raw logs to runs/ch03/.
"""
import argparse
import json
import os

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, latency, paths, provenance, vad  # noqa: E402
from hbasr.backends_sherpa import SherpaConfig, SherpaRecognizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{paths.BOOK_DATA}/runs/ch03"
WAV = f"{paths.BOOK_DATASETS}/librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0006.flac"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="zipformer-en")
ap.add_argument("--chunk-ms", type=float, default=160.0)
ap.add_argument("--wav", default=WAV)
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)

x, sr = audio.read_wav(args.wav)
speech = vad.onset_offset(x, sr)
cfg = SherpaConfig(model=args.model)
rec = SherpaRecognizer(cfg)  # one recognizer (model load is not part of any latency)
logs = {}
for pace in ("realtime", "simulated"):
    lg = latency.run_stream(rec, x, os.path.basename(args.wav), args.chunk_ms, pace, speech=speech)
    logs[pace] = lg
    open(os.path.join(OUT, f"log_{pace}.json"), "w").write(lg.to_json() + "\n")

sim = logs["simulated"]
print(f"speech onset {speech[0]:.3f} s, offset {speech[1]:.3f} s, audio {sim.audio_s:.3f} s (Silero VAD)")
print(f"{'chunk':>5} {'audio_end':>9} {'start':>7} {'done':>7} {'compute_ms':>10}  partial")
for e in sim.events[:12] + sim.events[-3:]:
    print(f"{e.index:5d} {e.audio_end_s:9.3f} {e.start_s:7.3f} {e.done_s:7.3f} {e.compute_s * 1000:10.2f}  {e.partial[-60:]}")
print(f"final at {sim.final_done_s:.3f} s: {sim.final_text}")
for pace, lg in logs.items():
    print(pace, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in lg.summary().items()}))


def ctx_ms(left, right, sub, hop_ms=10):
    frame = sub * hop_ms
    return {"left_ms": left * frame, "right_ms": right * frame, "chunk_ms_if_chunk=right+1": (right + 1) * frame}


table = []
for left, right in [(70, 13), (70, 6), (70, 1), (70, 0), (70, 16)]:
    for sub in (4, 8):
        table.append({"att_context_size": [left, right], "subsampling": sub} | ctx_ms(left, right, sub))
res = {
    "provenance": provenance.collect({"input": args.wav, "recognizer": cfg.to_dict()}),
    "speech_vad_s": speech,
    "summary": {p: lg.summary() for p, lg in logs.items()},
    "final_text": {p: lg.final_text for p, lg in logs.items()},
    "same_partials_both_pacings": [e.partial for e in logs["realtime"].events] == [e.partial for e in sim.events],
    "att_context_ms_table": table,
    "raw_logs": OUT,
}
provenance.write_json(os.path.join(HERE, "results", "harness_demo.json"), res)
print("same partial sequence in both pacings:", res["same_partials_both_pacings"])
for r in table:
    print(r)
