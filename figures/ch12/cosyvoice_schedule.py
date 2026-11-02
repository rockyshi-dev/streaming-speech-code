"""When CosyVoice 2's released Python code (cosyvoice/cli/model.py at 074ca6d) hands the first chunks to
flow matching, on the speech-token axis (25 tokens = 1 s): the paper's 15-token chunks against the
code's first hop of 25 tokens plus 0-24 padding tokens (to align with the prompt), then 3 look-ahead tokens, with later hops
of 50 and 100 tokens. The December 2024 code used 50-token hops."""
from figures.bookstyle import setup, figure, save, patch, synthetic

C, R = setup()
rows = [("paper (M = 15)", [(0, 15, R.current), (15, 3, R.lookahead), (18, 72, R.base)]),
        ("code, prompt pad 0", [(0, 25, R.current), (25, 3, R.lookahead), (28, 62, R.base)]),
        ("code, prompt pad 24", [(0, 49, R.current), (49, 3, R.lookahead), (52, 38, R.base)]),
        ("code, Dec 2024", [(0, 50, R.current), (50, 3, R.lookahead), (53, 37, R.base)])]
fig, ax = figure(aspect=0.36)
for y, (name, segs) in enumerate(reversed(rows)):
    for a, n, col in segs:
        ax.barh(y, n, 0.55, left=a, color=col, edgecolor=C.paper, lw=1)
ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 90)
ax.set_xlabel("speech tokens the language model has written (25 per second of audio)")
ax.legend(handles=[patch(R.current, "first chunk"), patch(R.lookahead, "3 look-ahead tokens"),
                   patch(R.base, "later chunks")],
          loc="upper left", bbox_to_anchor=(0, -0.3), ncols=3, fontsize=9, frameon=False)
ax.grid(axis="y", visible=False)
synthetic(ax)
save(fig)
