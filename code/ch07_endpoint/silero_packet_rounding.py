"""Chapter 7: Silero VAD's endpoint delays rounded to the 160 ms packets sherpa-onnx decides on.

    python code/ch07_endpoint/silero_packet_rounding.py      (after endpoint_sweep.py; ssh-asr env)

Silero decides on 32 ms windows; sherpa-onnx can decide only when a 160 ms client packet has been
processed. For a like-for-like comparison, each Silero decision time is moved up to the end of the
160 ms packet it falls in. English only, measured from the end of the last word in the forced
alignment (as in endpoint_vs_alignment.py), over the utterances that were not cut off early.
Writes results/silero_packet_rounding.json.
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import align_ref, paths, provenance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(f"{paths.BOOK_DATA}/runs/ch07/endpoint_sweep.jsonl")]
end = {}
for man in ["librispeech-test-clean-200", "librispeech-test-other-200"]:
    for u, ws in align_ref.load(man).items():
        end[u] = ws[-1][2]
by = defaultdict(list)
for r in rows:
    if r["lang"] == "en" and r["system"].startswith("silero") and r["utt_id"] in end:
        e = end[r["utt_id"]]
        if r["endpoint_s"] >= e - 0.05:  # not cut off early (same 50 ms rule as endpoint_sweep.py)
            by[r["setting"]].append((r["endpoint_s"] - e, math.ceil(r["endpoint_s"] / 0.16 - 1e-9) * 0.16 - e))
out = {"provenance": provenance.collect(), "min_silence_ms": {}}
for k, v in sorted(by.items(), key=lambda kv: float(kv[0])):
    a = np.array(v)
    out["min_silence_ms"][str(k)] = {"n": len(v), "p50_ms_32ms_windows": round(1000 * np.median(a[:, 0])),
                                     "p50_ms_rounded_to_160ms_packets": round(1000 * np.median(a[:, 1]))}
provenance.write_json(os.path.join(HERE, "results", "silero_packet_rounding.json"), out)
print(json.dumps(out["min_silence_ms"], indent=1))
