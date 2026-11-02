"""Qwen3-TTS on vLLM-Omni 0.32.0, one RTX 4090, open loop: sentence requests arriving as a Poisson process at a
fixed rate, sent whatever is still in flight. Time to first audio (p50, p95) against the arrival rate, with the
share of requests that underran. Shaded: rates above the server's measured capacity (bursts, no steady
state). Data: data/openloop.json, from code/serve/results/ch16_openloop*.json."""
import json
import pathlib

from figures.bookstyle import setup, figure, save

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "openloop.json"))["rows"]
x = [r["rate_per_s"] for r in d]
fig, ax = figure(aspect=0.45)
ax.plot(x, [r["ttfa_p50_s"] * 1000 for r in d], "o-", color=R.focus, label="TTFA p50")
ax.plot(x, [r["ttfa_p95_s"] * 1000 for r in d], "s--", color=R.base, label="TTFA p95")
ax.axvspan(14, max(x) + 1, color=C.muted, alpha=0.15, lw=0)
ax.text(14.3, 1330, "asked for more audio than\nthe server can produce", fontsize=8, va="top")
ax.axhline(1000, color=C.ink, lw=0.8, ls=":")
ax.text(0.6, 1030, "1 s target", fontsize=9)
ax2 = ax.twinx()
ax2.bar(x, [100 * r["requests_with_underrun"] / r["n"] for r in d], 0.6, color=R.lookahead, alpha=0.6)
ax2.set_ylim(0, 100)
ax2.set_ylabel("requests that underran (%)")
ax.set_xlabel("arrivals per second")
ax.set_ylabel("time to first audio (ms)")
ax.set_ylim(0, 1400)
ax.legend(fontsize=9, frameon=False, loc="upper left")
save(fig)
