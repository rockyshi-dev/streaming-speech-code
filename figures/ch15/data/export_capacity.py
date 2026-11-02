"""Export data/capacity.json for capacity.py from code/serve/results/ch15_*.json (code/serve/ch15/sweep.py):
partial-result latency p95 and clock drift at each number of concurrent streams, one series per server.

    python figures/ch15/data/export_capacity.py [--out PATH]
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "code" / "serve" / "results"
SERIES = [("CPU, sherpa-onnx defaults (320 ms model chunk)", "ch15_sherpa_cpu_b3t2"),
          ("CPU, sherpa-onnx tuned", "ch15_sherpa_cpu_b16t4p2"),
          ("GPU, Nemotron 80 ms chunks", "ch15_nemo_gpu_r0_fp32"),
          ("GPU, Nemotron 320 ms chunks", "ch15_nemo_gpu_r3_fp32"),
          ("GPU, Nemotron 560 ms chunks", "ch15_nemo_gpu_r6_fp32")]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "capacity.json"))
a = ap.parse_args()

series = {}
for label, name in SERIES:
    rows = json.load(open(RESULTS / f"{name}.json"))["rows"]
    series[label] = [{"streams": r["streams"], "partial_p95_ms": r["partial_latency_ms"]["p95"],
                      "drift_ms_per_s": r["slope_ms_per_s_median"]} for r in rows]
out = {"source": "code/serve/results/ch15_*.json (code/serve/ch15/sweep.py; i5-12600K and one RTX 4090)",
       "series": series}
json.dump(out, open(a.out, "w"), indent=1)
print({k: len(v) for k, v in series.items()}, "->", a.out)
