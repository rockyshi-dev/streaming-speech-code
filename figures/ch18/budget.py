"""The published 1,293 ms voice-to-voice budget (Daily/Pipecat guide, section 5.1, fetched 2026-10-29),
grouped by owner: speech recognition and synthesis, the language model, transport and device."""
from figures.bookstyle import setup, figure, save, patch

C, R = setup()
rows = [("mic", 40, "t"), ("enc", 21, "t"), ("net", 10, "t"), ("pkt", 2, "t"), ("jitter", 40, "t"), ("dec", 1, "t"),
        ("ASR + endpoint", 300, "a"), ("LLM first byte", 650, "l"), ("sentence", 20, "s"),
        ("TTS", 120, "s"), ("enc", 21, "t"), ("pkt", 2, "t"), ("net", 10, "t"), ("jitter", 40, "t"),
        ("dec", 1, "t"), ("speaker", 15, "t")]
col = {"t": R.lat.transport, "a": R.sys.asr, "l": R.sys.llm, "s": R.sys.tts}
fig, ax = figure("wide", aspect=0.2)
x = 0
for name, ms, k in rows:
    ax.barh(0, ms, 0.6, left=x, color=col[k], edgecolor=C.paper)
    if ms >= 100:
        ax.text(x + ms / 2, 0, f"{name}\n{ms} ms", ha="center", va="center", fontsize=9, color=C.paper)
    x += ms
ax.text(x + 10, 0, f"{x:,} ms", va="center", fontsize=10)
ax.set_yticks([])
ax.set_xlim(0, 1420)
ax.set_xlabel("ms after the user stops speaking")
ax.grid(axis="y", visible=False)
ax.legend(handles=[patch(R.sys.asr, "speech recognition (300)"), patch(R.sys.llm, "language model (650)"),
                   patch(R.sys.tts, "synthesis incl. sentence (140)"), patch(R.lat.transport, "transport and device (203)")],
          loc="upper left", bbox_to_anchor=(0, -0.55), ncols=4, handlelength=1.0, fontsize=9)
save(fig)
