"""Export data/underrun.json for underrun.py: one paragraph that Kokoro synthesized sentence by sentence and
the player ran dry on. The per-sentence timings are in the raw rows of code/tts/ch12_kokoro_cpu/kokoro_stream.py
($BOOK_DATA/runs/ch12_kokoro_cpu/rows.jsonl), not in its summary, so this exporter reads those rows.

    python figures/ch14/data/export_underrun.py [--rows PATH] [--out PATH]
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "code" / "tts"))
from hbtts import paths  # noqa: E402

PARAGRAPH = "7176_92135_000031_000003+7176_92135_000031_000004+7176_92135_000031_000006"
ap = argparse.ArgumentParser()
ap.add_argument("--rows", default=str(paths.book_data("runs", "ch12_kokoro_cpu", "rows.jsonl")))
ap.add_argument("--out", default=str(HERE / "underrun.json"))
a = ap.parse_args()

rows = [json.loads(line) for line in open(a.rows, encoding="utf-8") if line.strip()]
r = next(x for x in rows if x["id"] == PARAGRAPH and x["config"] == "en" and x["mode"] == "sentence"
         and x["repeat"] == 0)
out = {
    "source": "raw rows of code/tts/ch12_kokoro_cpu/kokoro_stream.py (Kokoro-82M v1.0, en, sentence mode, 4 CPU "
              "threads), first of three runs of this paragraph",
    "id": r["id"],
    "ttfa_s": r["ttfa_s"],
    "underruns": r["underruns"],
    "stall_s": r["stall_s"],
    # a sentence's audio is ready when its synthesis ends
    "chunks": [{"text": c["text"], "ready_s": c["synth_start_s"] + c["synth_s"], "audio_s": c["audio_s"]}
               for c in r["chunks"]],
}
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
print(len(out["chunks"]), "sentences ->", a.out)
