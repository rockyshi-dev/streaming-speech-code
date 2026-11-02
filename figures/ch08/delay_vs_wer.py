"""Delay against WER for a wrapper and for a native streaming model, each on its own benchmark (the
two panels are not comparable with each other). Left: Whisper-Streaming, large-v2 with faster-whisper
on an A40, English ESIC dev, computationally aware latency per minimum chunk size (arXiv:2307.14743,
Table 3). Right: Voxtral Realtime, En-Short macro average, per configured delay (arXiv:2602.11298,
Table 3), with offline Whisper on the same sets."""
from figures.bookstyle import setup, figure, save

C, R = setup()
fig, (a, b) = figure("wide", aspect=0.38, ncols=2)
ws = [(0.5, 3270, 8.5), (1.0, 3620, 8.1), (2.0, 5450, 8.0)]
a.plot([x[1] for x in ws], [x[2] for x in ws], "o-", color=R.focus)
for c, lat, w in ws:
    a.annotate(f"chunk {c:g} s", (lat, w), textcoords="offset points", xytext=(6, 6), fontsize=9)
a.axhline(7.9, color=C.ink, lw=0.8, ls=":")
a.text(3300, 7.93, "offline Whisper: 7.9", fontsize=9, va="bottom")
a.set_title("Whisper-Streaming (wrapper)", loc="left", fontsize=10)
a.set_xlabel("average latency (ms)")
a.set_ylabel("WER (%)")
a.set_xlim(3000, 6200)
a.set_ylim(7.6, 8.9)
vx = [(240, 9.95), (480, 8.47), (2400, 7.72)]
b.plot([x[0] for x in vx], [x[1] for x in vx], "o-", color=R.focus)
for d, w in vx:
    b.annotate(f"delay {d:,} ms", (d, w), textcoords="offset points", xytext=(6, 6), fontsize=9)
b.axhline(8.39, color=C.ink, lw=0.8, ls=":")
b.text(1300, 8.42, "offline Whisper: 8.39", fontsize=9, va="bottom")
b.set_title("Voxtral Realtime (native)", loc="left", fontsize=10)
b.set_xlabel("configured delay (ms)")
b.set_xlim(0, 3000)
b.set_ylim(7.4, 10.4)
save(fig)
