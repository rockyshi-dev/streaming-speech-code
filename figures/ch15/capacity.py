"""Partial-result latency (p95) against concurrent real-time streams for the CPU track's sherpa-onnx server
(defaults and tuned) and the GPU track's Nemotron 3.5 ASR server at three trained chunk sizes, on one
i5-12600K and one RTX 4090. The last point of each line is the first load at which the server fell behind.
Data: data/capacity.json, from code/serve/results/ch15_*.json."""
import json
import pathlib

from figures.bookstyle import setup, figure, save

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "capacity.json"))["series"]
fig, ax = figure("wide", aspect=0.42)
styles = [("o-", C.series[0]), ("o--", C.series[0]), ("s-", C.series[1]), ("s-", C.series[2]), ("s-", C.series[3])]
for (name, rows), (m, col) in zip(d.items(), styles):
    ax.plot([r["streams"] for r in rows], [r["partial_p95_ms"] for r in rows], m, color=col, label=name, ms=5)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([1, 4, 16, 64, 256, 512], ["1", "4", "16", "64", "256", "512"])
ax.set_xlabel("concurrent real-time streams")
ax.set_ylabel("partial-result latency, p95 (ms)")
ax.legend(fontsize=8, frameon=False, loc="upper left")
save(fig)
