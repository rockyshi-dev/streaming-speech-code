"""Qwen3-TTS (12 Hz, 1.7B CustomVoice) served by vLLM-Omni 0.32.0 on one RTX 4090: time to first audio for
10 three-sentence paragraphs per language, with the text sent at once (HTTP) or streamed in at 20 words or
characters per second over the WebSocket with no splitting, sentence splitting or commitment mode. Bars: p50;
whiskers: p95. Data: data/qwen3tts_modes.json, from code/tts/results/ch12_qwen3tts_input_modes.json."""
import json
import pathlib

from figures.bookstyle import setup, figure, save, patch

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "qwen3tts_modes.json"))
cells = [("http", None, "all text at once (HTTP)", R.base), ("ws-none", 20.0, "streamed in, no split", R.muted),
         ("ws-sentence", 20.0, "streamed in, sentence split", R.focus), ("ws-commit", 20.0, "streamed in, commitment", R.lookahead)]
fig, ax = figure(aspect=0.42)
w = 0.2
for i, lang in enumerate(["en", "zh"]):
    for j, (mode, rate, name, col) in enumerate(cells):
        r = next(x for x in d["rows"] if x["mode"] == mode and x["text_rate_units_per_s"] == rate and x["lang"] == lang)
        x = i + (j - 1.5) * w
        ax.bar(x, r["ttfa_s"]["p50"] * 1000, w * 0.9, color=col)
        ax.plot([x, x], [r["ttfa_s"]["p50"] * 1000, r["ttfa_s"]["p95"] * 1000], color=C.ink, lw=1)
        ax.text(x, r["ttfa_s"]["p95"] * 1000 + 80, f"{r['ttfa_s']['p50'] * 1000:,.0f}", ha="center", fontsize=8)
ax.set_xticks([0, 1], ["English", "Mandarin"])
ax.set_ylabel("time to first audio (ms)")
ax.set_ylim(0, 4300)
ax.legend(handles=[patch(c, n) for _, _, n, c in cells], fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncols=4)
ax.grid(axis="x", visible=False)
save(fig)
