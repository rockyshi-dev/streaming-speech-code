"""A jitter buffer on a synthetic network: 20 ms packets sent on a fixed schedule arrive with random
jitter; playout runs a fixed target delay behind the send time; a packet that arrives after its
playout time is concealed. Synthetic data, fixed seed."""
import random

from figures.bookstyle import setup, figure, save, patch, synthetic

C, R = setup()
rng = random.Random(7)
send = [20 * i for i in range(16)]
arrive = [t + 30 + max(0.0, rng.gauss(0, 14)) for t in send]
arrive[9] += 45  # one late packet
target = 60
fig, ax = figure(aspect=0.42)
for i, (s, a) in enumerate(zip(send, arrive)):
    play = s + target
    late = a > play
    ax.plot([s, a], [1, 0.5], color=R.base, lw=0.8)
    ax.scatter([a], [0.5], s=14, color=R.focus if late else R.lat.transport, zorder=3)
    ax.barh(0, 20, 0.25, left=play, color=R.focus if late else R.sys.tts, edgecolor=C.paper)
ax.set_yticks([1, 0.5, 0], ["sent", "arrives", "played"])
ax.set_xlabel("time (ms)")
ax.set_xlim(-5, send[-1] + target + 30)
ax.grid(axis="y", visible=False)
ax.legend(handles=[patch(R.sys.tts, "played on time"), patch(R.focus, "too late: concealed")],
          loc="upper left", bbox_to_anchor=(0, -0.28), ncols=2, handlelength=1.0)
synthetic(ax)
save(fig)
