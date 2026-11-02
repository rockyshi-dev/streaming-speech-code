"""The device and network share of one voice-to-voice turn, from the Daily/Pipecat budget table
(voiceaiandvoiceagents.com, fetched 2026-10-29): inbound 114 ms, outbound 89 ms."""
from figures.bookstyle import setup, figure, save

C, R = setup()
inbound = [("microphone", 40), ("Opus encode", 21), ("network", 10), ("packet handling", 2),
           ("jitter buffer", 40), ("decode", 1)]
outbound = [("Opus encode", 21), ("packet handling", 2), ("network", 10), ("jitter buffer", 40),
            ("decode", 1), ("speaker", 15)]
fig, ax = figure(aspect=0.3)
for y, rows in [(1, inbound), (0, outbound)]:
    x = 0
    for name, ms in rows:
        col = R.lat.transport if name in ("network", "packet handling", "jitter buffer") else R.base
        ax.barh(y, ms, 0.5, left=x, color=col, edgecolor=C.paper)
        if ms >= 10:
            ax.text(x + ms / 2, y, name, ha="center", va="center", fontsize=8, rotation=0)
        x += ms
    ax.text(x + 3, y, f"{x} ms", va="center", fontsize=9)
ax.set_yticks([1, 0], ["device to server", "server to device"])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 140)
ax.set_xlabel("ms, from the published budget table")
ax.grid(axis="y", visible=False)
save(fig)
