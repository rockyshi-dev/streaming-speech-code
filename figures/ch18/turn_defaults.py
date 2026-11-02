"""Turn-detection defaults on one ms axis, from code: Pipecat v1.15.0 (VAD stop 0.2 s, Smart Turn v3
with a 3 s silence fallback) and LiveKit Agents 1.8.10 (Silero min silence 0.55 s; endpointing
min/max 0.5/3.0 s, or 0.3/2.5 s with a streaming turn detector)."""
from figures.bookstyle import setup, figure, save

C, R = setup()
rows = [("Pipecat: VAD stop", 0, 200, "then Smart Turn decides (fallback 3 s)"),
        ("LiveKit: Silero min silence", 0, 550, ""),
        ("LiveKit endpointing (default)", 500, 3000, "min 0.5 s, max 3.0 s"),
        ("LiveKit endpointing (turn model)", 300, 2500, "min 0.3 s, max 2.5 s")]
fig, ax = figure(aspect=0.4)
for y, (name, a, b, note) in enumerate(reversed(rows)):
    ax.barh(y, b - a, 0.5, left=a, color=R.lat.endpoint if a == 0 else R.base)
    if note:
        ax.text(b + 40, y, note, va="center", fontsize=9)
ax.axvline(300, color=C.ink, lw=0.8, ls=":")
ax.text(310, len(rows) - 0.45, "budget row: 300 ms", fontsize=9)
ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 4400)
ax.set_xlabel("silence after the user's last word (ms)")
ax.grid(axis="y", visible=False)
save(fig)
