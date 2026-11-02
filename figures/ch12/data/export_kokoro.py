"""Export data/kokoro.json for kokoro_modes.py from code/tts/results/ch12_kokoro_cpu.json
(code/tts/ch12_kokoro_cpu/kokoro_stream.py).

    python figures/ch12/data/export_kokoro.py [--out PATH]
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "kokoro.json"))
a = ap.parse_args()

r = json.load(open(REPO / "code" / "tts" / "results" / "ch12_kokoro_cpu.json"))


def pct(x):
    return {"p50": x["p50"], "p95": x["p95"], "n": x["n"]}


out = {
    "source": "code/tts/results/ch12_kokoro_cpu.json (code/tts/ch12_kokoro_cpu/kokoro_stream.py --threads 4, i5-12600K)",
    "paragraphs": [{"config": p["config"], "mode": p["mode"], "rate": p["rate"], "n": p["n"], "ttfa_s": pct(p["ttfa_s"]),
                    "rtf": pct(p["rtf"]), "underruns_total": p["underruns_total"], "stall_s": pct(p["stall_s"])}
                   for p in r["paragraphs"]],
}
json.dump(out, open(a.out, "w"), indent=1)
print(len(out["paragraphs"]), "rows ->", a.out)
