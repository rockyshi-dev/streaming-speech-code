"""Export data/measured.json for measured_wer_vs_latency.py from code/ch09_measure/results/summary_table.json
(code/ch09_measure/make_tables.py writes that table).

    python figures/ch09/data/export_measured.py [--out PATH]
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
KEYS = ("model", "decoder", "att_context_size", "manifest", "error_rate_pct", "error_rate_raw_pct", "UPWR",
        "first_token_p50_ms", "first_token_p90_ms", "final_after_speech_end_p50_ms", "word_latency_fa_p50_ms",
        "word_latency_fa_p90_ms", "corpus_rtf", "utterances")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "measured.json"))
a = ap.parse_args()

table = json.load(open(REPO / "code" / "ch09_measure" / "results" / "summary_table.json"))
out = {"source": "code/ch09_measure/results/summary_table.json", "rows": [{k: r.get(k) for k in KEYS} for r in table]}
json.dump(out, open(a.out, "w"), indent=1)
print(len(out["rows"]), "rows ->", a.out)
