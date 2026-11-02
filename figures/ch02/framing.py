"""Where the first five 25 ms windows sit, from Kaldi feature-window.cc: snip_edges=true starts
frame f at sample 160 f; snip_edges=false centres it on 160 f + 80, covering [160 f - 120, 160 f + 280)
and reflecting the signal before sample 0. Frame f ends 15 ms (snip_edges=true) or 7.5 ms (false) after its
10 ms hop does."""
import numpy as np

from figures.bookstyle import setup, figure, save

C, R = setup()
fig, ax = figure(aspect=0.42)
rows = [("snip_edges=true", 0), ("snip_edges=false", -7.5)]
for k, (name, off) in enumerate(rows):
    for f in range(5):
        y = k * 6.5 + f
        a = 10 * f + off
        ax.barh(y, 25, 0.7, left=a, color=R.base if f else R.focus)
        ax.barh(y, 10, 0.7, left=10 * f, fill=False, edgecolor=C.ink, lw=0.8)
    ax.text(-12, k * 6.5 + 2, name, ha="right", va="center", fontsize=10)
ax.axvline(0, color=C.ink, lw=0.8)
ax.text(1, -1.2, "first sample", fontsize=9)
ax.set_yticks([])
ax.set_xlim(-45, 70)
ax.set_ylim(-2, 11.5)
ax.set_xlabel("time (ms)")
ax.grid(axis="y", visible=False)
save(fig)
