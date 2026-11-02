"""Export the data for timeline.py from the real-time harness log of chapter 3
(code/ch03_latency/harness_demo.py writes it to $BOOK_DATA/runs/ch03/).

    python figures/ch03/data/export_timeline.py [path/to/log_realtime.json]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "code"))
from hbasr import paths  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else f"{paths.BOOK_DATA}/runs/ch03/log_realtime.json"
d = json.load(open(src))
here = os.path.dirname(os.path.abspath(__file__))
summary = json.load(open(os.path.join(here, "..", "..", "..", "code", "ch03_latency", "results", "harness_demo.json")))
out = {
    "source": "code/ch03_latency/harness_demo.py, real-time pacing, " + d["utt_id"],
    "chunk_ms": d["chunk_ms"],
    "audio_s": d["audio_s"],
    "speech_vad_s": summary["speech_vad_s"],
    "final_done_s": d["final_done_s"],
    "final_words": len(d["final_text"].split()),
    "chunks": [{"arrival_s": round(e["arrival_s"], 4), "done_s": round(e["done_s"], 4),
                "compute_ms": round(1000 * e["compute_s"], 2), "words": len(e["partial"].split())}
               for e in d["events"]],
}
json.dump(out, open(os.path.join(here, "timeline.json"), "w"), indent=1)
print(len(out["chunks"]), "chunks")
