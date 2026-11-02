"""Export data/boundaries.json for vocos_joins.py from code/tts/results/ch13_chunk_boundaries.json
(code/tts/ch13_chunk_boundaries/chunk_boundaries.py).

    python figures/ch13/data/export_boundaries.py [--out PATH]
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
KEYS = ("part", "chunk_frames", "method", "added_latency_ms", "boundary_err_db_median_mean_over_clips",
        "boundary_err_db_max_mean_over_clips", "interior_err_db_median_mean_over_clips", "snr_db_mean_over_clips")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "boundaries.json"))
a = ap.parse_args()

r = json.load(open(REPO / "code" / "tts" / "results" / "ch13_chunk_boundaries.json"))
out = {
    "source": "code/tts/results/ch13_chunk_boundaries.json (code/tts/ch13_chunk_boundaries/chunk_boundaries.py)",
    "vocos_receptive_field_measured": r["vocos_receptive_field_measured"],
    "rows": [{k: t.get(k) for k in KEYS} for t in r["table"]],
}
json.dump(out, open(a.out, "w"), indent=1)
print(len(out["rows"]), "rows ->", a.out)
