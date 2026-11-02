"""Which frames each feature normalizer reads when it normalizes frame t (here t = 3 s of a 5 s
utterance), from the code cited in chapter 2: per-utterance mean and variance (NeMo per_feature,
Kaldi offline CMVN) read every frame; Kaldi online CMVN reads the previous 600 frames; a fixed
global CMVN (WeNet) or none (NeMo streaming, icefall) reads no frame of this utterance. Schematic."""
from figures.bookstyle import setup, figure, save, patch

C, R = setup()
T, t = 10000, 8000
rows = [("per-utterance (NeMo per_feature)", [(0, t, R.past), (t, 80, R.current), (t + 80, T - t - 80, R.lookahead)]),
        ("online, last 600 frames (Kaldi)", [(t - 6000, 6000, R.past), (t, 80, R.current)]),
        ("global constants or none (WeNet, icefall, NeMo streaming)", [(t, 80, R.current)])]
fig, ax = figure(aspect=0.3)
for y, (name, segs) in enumerate(reversed(rows)):
    ax.barh(y, T, 0.55, left=0, color=C.paper, edgecolor=C.line, lw=0.8)
    for a, w, col in segs:
        ax.barh(y, w, 0.55, left=a, color=col)
ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, T)
ax.set_xlabel("time in a 10 s utterance (ms); the frame being normalized is at 8,000 ms")
ax.legend(handles=[patch(R.past, "past frames read"), patch(R.current, "this frame"),
                   patch(R.lookahead, "future frames read")], loc="upper left", bbox_to_anchor=(0, -0.45),
          ncols=3, fontsize=9, frameon=False)
ax.grid(axis="y", visible=False)
save(fig)
