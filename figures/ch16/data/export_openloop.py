"""Export data/openloop.json for knee.py from code/serve/results/ch16_openloop.json and ch16_openloop_high.json
(code/serve/ch16/tts_openloop.py openloop: the lower and the higher arrival rates).

    python figures/ch16/data/export_openloop.py [--out PATH]
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "code" / "serve" / "results"
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "openloop.json"))
a = ap.parse_args()

rows = []
for name in ("ch16_openloop", "ch16_openloop_high"):
    for t in json.load(open(RESULTS / f"{name}.json"))["table"]:
        rows.append({"rate_per_s": t["rate_per_s"], "n": t["n"], "ttfa_p50_s": t["ttfa_s"]["p50"],
                     "ttfa_p95_s": t["ttfa_s"]["p95"], "rtf_p50": t["rtf"]["p50"],
                     "in_flight_p95": t["in_flight_at_arrival"]["p95"],
                     "requests_with_underrun": t["requests_with_underrun"]})
out = {"source": "code/serve/results/ch16_openloop.json and ch16_openloop_high.json (code/serve/ch16/tts_openloop.py, "
                 "one RTX 4090)",
       "rows": sorted(rows, key=lambda r: r["rate_per_s"])}
json.dump(out, open(a.out, "w"), indent=1)
print(len(out["rows"]), "rows ->", a.out)
