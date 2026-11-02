"""Delay-penalized transducer, streaming Conformer, LibriSpeech test-clean, 640 ms chunks
(arXiv:2211.00490, Table 1): mean emission delay against WER for lambda = 0, 0.003, 0.006, 0.01.
Published numbers, plotted as given."""
from figures.bookstyle import setup, figure, save

C, R = setup()
pts = [(0, 3.40, 373), (0.003, 3.70, 102), (0.006, 3.74, -3), (0.01, 4.67, -93)]
fig, ax = figure(aspect=0.5)
ax.plot([p[2] for p in pts], [p[1] for p in pts], color=R.lat.emission, lw=1.4, marker="o")
for lam, wer, d in pts:
    ax.annotate(f"λ = {lam}", (d, wer), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.axvline(0, color=C.muted, lw=0.8, ls=":")
ax.set_xlabel("mean emission delay (ms), incl. half the 640 ms chunk")
ax.set_ylabel("WER, test-clean (%)")
ax.invert_xaxis()
save(fig)
