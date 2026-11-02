"""Vocos (24 kHz, 100-bin mel, hop 256 = 10.67 ms) decoding the same mel in chunks of 16 frames
(171 ms) with different amounts of left context, right context and linear crossfade, against
decoding it in one call: median error energy in a 5 ms window at each join, relative to the signal,
averaged over four clips. 0 dB means the error is as loud as the speech; below -60 dB is inaudible.
Data: data/boundaries.json, from code/tts/results/ch13_chunk_boundaries.json."""
import json
import pathlib

from figures.bookstyle import setup, figure, save

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "boundaries.json"))
names = {"naive": "no context", "xfade2": "crossfade 2", "left_ctx8": "left 8", "left8_right4": "left 8, right 4",
         "left8_right2_xfade2": "left 8, right 2,\ncrossfade 2", "left16_right8_xfade4": "left 16, right 8,\ncrossfade 4",
         "left32_right32": "left 32, right 32"}
order = ["naive", "xfade2", "left_ctx8", "left8_right4", "left8_right2_xfade2", "left16_right8_xfade4",
         "left32_right32"]
pts = {r["method"]: r for r in d["rows"] if r["part"] == "A_vocos" and r["chunk_frames"] == 16}
fig, ax = figure(aspect=0.5)
for y, m in enumerate(reversed(order)):
    r = pts[m]
    v = r["boundary_err_db_median_mean_over_clips"]
    ax.barh(y, v, 0.55, color=R.focus if m == "left32_right32" else R.base)
    ax.text(v - 2 if v < 0 else 2, y, f"+{r['added_latency_ms']:.0f} ms", va="center",
            ha="right" if v < 0 else "left", fontsize=8)
ax.set_yticks(range(len(order)), [names[m].replace("\n", " ") for m in reversed(order)])
ax.tick_params(axis="y", length=0)
ax.axvline(0, color=C.ink, lw=0.8)
ax.set_xlabel("median error at the joins (dB relative to the signal); label: added latency")
ax.set_xlim(-135, 12)
ax.grid(axis="y", visible=False)
save(fig)
