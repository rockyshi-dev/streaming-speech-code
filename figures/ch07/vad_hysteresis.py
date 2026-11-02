"""Silero VAD on one utterance (LibriSpeech test-clean 1089-134686-0006 plus 1 s of silence):
speech probability per 32 ms window, the start (0.5) and end (0.35) thresholds, and the segments
VADIterator reports with its defaults. The pause at 7.3 s inside the sentence is reported as an end
of speech. Data: data/vad.json, exported by data/export_vad.py with silero-vad 6.2.3."""
import json
import pathlib

import numpy as np

from figures.bookstyle import setup, figure, save

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "vad.json"))
p = np.array(d["probs"])
t = (np.arange(p.size) + 1) * d["window_ms"]
fig, ax = figure("wide", aspect=0.36)
ax.step(t, p, where="post", color=R.base, lw=1.0)
ax.axhline(0.5, color=C.ink, lw=0.8, ls=":")
ax.axhline(0.35, color=C.ink, lw=0.8, ls="--")
ax.text(t[-1] + 60, 0.5, "start above 0.5", va="center", fontsize=9)
ax.text(t[-1] + 60, 0.35, "end below 0.35\nfor 100 ms", va="center", fontsize=9)
ev = d["events"]
starts = [e for e in ev if "start" in e]
ends = [e for e in ev if "end" in e]
for s, e in zip(starts, ends):
    ax.plot([s["start"] * 1000, e["end"] * 1000], [1.1, 1.1], color=R.focus, lw=5, solid_capstyle="butt")
for e in ends:
    ax.plot(e["window_end_s"] * 1000, 1.1, "v", color=C.ink, ms=6)
ax.text(starts[0]["start"] * 1000, 1.2, "segments reported (triangle: when the end was decided)", fontsize=9)
ax.set_ylim(-0.05, 1.32)
ax.set_xlim(0, t[-1])
ax.set_xlabel("time (ms)")
ax.set_ylabel("speech probability")
save(fig)
