"""Qwen3-TTS (12 Hz, 1.7B CustomVoice) on vLLM-Omni 0.32.0, one RTX 4090, closed loop of N clients each sending
sentence-length requests: time to first audio (p50, p95), real-time factor (p50) and the share of requests that
underran, against N. Data: data/concurrency.json, from code/tts/results/ch14_qwen3tts_concurrency.json."""
import json
import pathlib

from figures.bookstyle import setup, figure, save

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "concurrency.json"))["rows"]
n = [r["concurrency"] for r in d]
fig, (a, b) = figure("wide", aspect=0.36, ncols=2)
a.plot(n, [r["ttfa_p50_s"] * 1000 for r in d], "o-", color=R.focus, label="p50")
a.plot(n, [r["ttfa_p95_s"] * 1000 for r in d], "s--", color=R.base, label="p95")
a.set_xscale("log", base=2)
a.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
a.set_xlabel("concurrent requests")
a.set_ylabel("time to first audio (ms)")
a.legend(fontsize=9, frameon=False)
b.plot(n, [r["rtf_p50"] for r in d], "o-", color=R.focus, label="RTF p50")
b.axhline(1, color=C.ink, lw=0.8, ls=":")
b2 = b.twinx()
b2.bar(n, [100 * r["requests_with_underrun"] / r["n"] for r in d], [x * 0.25 for x in n], color=R.lookahead, alpha=0.6)
b2.set_ylabel("requests that underran (%)")
b2.set_ylim(0, 100)
b.set_xscale("log", base=2)
b.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
b.set_xlabel("concurrent requests")
b.set_ylabel("real-time factor, p50")
b.set_ylim(0, 1.4)
save(fig)
