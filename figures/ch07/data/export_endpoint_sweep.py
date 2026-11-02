"""Export data/endpoint_sweep.json for endpoint_tradeoff.py from code/ch07_endpoint/results/endpoint_sweep.json
(code/ch07_endpoint/endpoint_sweep.py, then endpoint_vs_alignment.py for the English forced-alignment block).
English is read against the end of the last word in a forced alignment, Mandarin against the offline VAD.

    python figures/ch07/data/export_endpoint_sweep.py [--out PATH]
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "endpoint_sweep.json"))
a = ap.parse_args()

r = json.load(open(REPO / "code" / "ch07_endpoint" / "results" / "endpoint_sweep.json"))


def rows(block):
    return [{"setting": s, "p50_ms": round(1000 * v["endpoint_delay_s"]["p50"]), "false_cutoffs": v["false_cutoffs"],
             "utterances": v["utterances"]} for s, v in block.items()]


fa = r["en_vs_forced_alignment"]
out = {
    "source": "code/ch07_endpoint/results/endpoint_sweep.json (code/ch07_endpoint/endpoint_sweep.py, 100 utterances "
              "per language)",
    "en_forced_alignment": {"sherpa_rule2_s": rows(fa["sherpa-rule2"]), "silero_min_silence_ms": rows(fa["silero"])},
    "zh_vad_reference": {"sherpa_rule2_s": rows(r["sherpa"]["zh"]), "silero_min_silence_ms": rows(r["silero"]["zh"])},
    "material": r["material"],
}
json.dump(out, open(a.out, "w"), indent=1)
print("->", a.out)
