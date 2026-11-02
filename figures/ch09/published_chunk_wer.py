"""Published accuracy against chunk size for three streaming model families, each on its own test set
and with its own chunk convention (so the panels must not be compared with each other):
icefall streaming Zipformer, LibriSpeech test-other, greedy, chunk-wise decoding
(egs/librispeech/ASR/RESULTS.md lines 821-834 at 9c96676); WeNet unified Conformer, AISHELL-1 test
CER, attention rescoring, chunk in 40 ms frames (examples/aishell/s0/README.md lines 48-61 at
be6ad97); Nemotron 3.5 ASR streaming, English (en-US, en-GB) WER, chunk = (right + 1) x 80 ms
(model card README line 696 at ea30d66)."""
from figures.bookstyle import setup, figure, save

C, R = setup()
panels = [("icefall Zipformer\nLibriSpeech test-other WER", [(320, 7.79), (640, 7.16)]),
          ("WeNet unified Conformer\nattention rescoring\nAISHELL-1 test CER", [(160, 5.91), (320, 5.69), (640, 5.45)]),
          ("Nemotron 3.5 ASR\nEnglish WER", [(80, 9.43), (160, 8.88), (320, 8.27), (560, 7.99), (1120, 7.91)])]
fig, axes = figure("wide", aspect=0.34, ncols=3)
for ax, (title, pts) in zip(axes, panels):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=R.focus)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("chunk (ms)")
    lo, hi = min(p[1] for p in pts), max(p[1] for p in pts)
    ax.set_ylim(lo - 0.3 * (hi - lo) - 0.1, hi + 0.3 * (hi - lo) + 0.1)
    ax.set_xlim(0, max(p[0] for p in pts) * 1.15)
axes[0].set_ylabel("error rate (%)")
save(fig)
