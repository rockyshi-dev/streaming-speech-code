"""Re-score existing raw logs with the current scoring code (no audio is streamed again).

    python code/ch09_measure/rescore.py [results/*.json ...]

Reads each summary's `raw` path (a `$BOOK_DATA/...` placeholder is expanded) and `provenance.manifest`, recomputes accuracy, stability and latency
(including the forced-alignment word latency once runs/align/<manifest>.json exists), keeps the
original provenance and adds a `rescored` block.
"""
import glob
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import evaluate, provenance  # noqa: E402
from hbasr import paths as bookpaths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
paths = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "results", "*.json")))
for p in paths:
    import json

    d = json.load(open(p))
    if "accuracy" not in d or "raw" not in d:
        continue
    new = evaluate.score(bookpaths.resolve(d["raw"]), d["lang"], d["provenance"].get("manifest"))
    new["provenance"] = d["provenance"]
    new["rescored"] = provenance.collect()
    provenance.write_json(p, new)
    w = new.get("word_emission_latency_vs_forced_alignment_s", {})
    print(os.path.basename(p), "word latency p50", w.get("p50"), "p90", w.get("p90"))
