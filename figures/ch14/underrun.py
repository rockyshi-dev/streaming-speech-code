"""One measured underrun: a three-sentence paragraph synthesized sentence by sentence by Kokoro-82M on
four CPU threads. Staircase: seconds of audio the player has received; line: seconds of audio the
player needs, starting at the first audio and running in real time. Where the line rises above the
staircase the player has nothing to play. Data: data/underrun.json (Kokoro run of chapter 12)."""
import json
import pathlib

import numpy as np

from figures.bookstyle import setup, figure, save

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "underrun.json"))
ch = d["chunks"]
k = 1000.0
fig, ax = figure(aspect=0.45)
got, t_prev = 0.0, 0.0
xs, ys = [0.0], [0.0]
for c in ch:
    xs += [c["ready_s"] * k, c["ready_s"] * k]
    ys += [got * k, (got + c["audio_s"]) * k]
    got += c["audio_s"]
xs.append(8000)
ys.append(got * k)
ax.plot(xs, ys, color=R.base, lw=1.6, label="audio received")
# the player: starts at the first chunk, waits when it runs out
start = ch[0]["ready_s"]
t, played = start, 0.0
px, py = [start * k], [0.0]
avail = 0.0
for c in ch:
    if c["ready_s"] > t:
        px.append(c["ready_s"] * k)
        py.append(played * k)
        t = c["ready_s"]
    played += c["audio_s"]
    t += c["audio_s"]
    if t * k > 8000:
        frac = (8000 / k - (t - c["audio_s"]))
        px.append(8000)
        py.append((played - c["audio_s"] + frac) * k)
        break
    px.append(t * k)
    py.append(played * k)
ax.plot(px, py, color=R.focus, lw=1.6, label="audio the player needs")
gap0 = ch[0]["ready_s"] + ch[0]["audio_s"]
ax.axvspan(gap0 * k, ch[1]["ready_s"] * k, color=R.focus, alpha=0.15, lw=0)
ax.text((gap0 + ch[1]["ready_s"]) / 2 * k, 6000, f"underrun\n{(ch[1]['ready_s'] - gap0) * k:,.0f} ms",
        ha="center", fontsize=9)
ax.set_xlim(0, 8000)
ax.set_ylim(0, 21000)
ax.set_xlabel("time since the request (ms)")
ax.set_ylabel("audio (ms)")
ax.legend(fontsize=9, frameon=False, loc="upper left")
save(fig)
