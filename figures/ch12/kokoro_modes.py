"""Kokoro-82M on four CPU threads (i5-12600K): time to first audio for 5 three-sentence paragraphs (3 runs each) per
language, synthesized as one call (whole), sentence by sentence with all text known (sentence), and
sentence by sentence with text arriving at 20 words or characters per second (stream). Bars: p50;
whiskers: p95. Data: data/kokoro.json, from code/tts/results/ch12_kokoro_cpu.json."""
import json
import pathlib

import numpy as np

from figures.bookstyle import setup, figure, save, patch

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "kokoro.json"))
rows = {(p["config"], p["mode"]): p for p in d["paragraphs"]}
langs = [("en", "English (v1.0)"), ("zh", "Mandarin (v1.1-zh)")]
modes = [("whole", R.base), ("sentence", R.focus), ("stream", R.lookahead)]
fig, ax = figure(aspect=0.42)
w = 0.26
for i, (lang, _) in enumerate(langs):
    for j, (mode, col) in enumerate(modes):
        p = rows[(lang, mode)]["ttfa_s"]
        x = i + (j - 1) * w
        ax.bar(x, p["p50"] * 1000, w * 0.9, color=col)
        ax.plot([x, x], [p["p50"] * 1000, p["p95"] * 1000], color=C.ink, lw=1)
        ax.text(x, p["p95"] * 1000 + 120, f"{p['p50'] * 1000:,.0f}", ha="center", fontsize=8)
ax.set_xticks(range(len(langs)), [l[1] for l in langs])
ax.set_ylabel("time to first audio (ms)")
ax.set_ylim(0, 8200)
ax.legend(handles=[patch(c, m) for m, c in modes], fontsize=9, frameon=False, loc="upper right")
ax.grid(axis="x", visible=False)
save(fig)
