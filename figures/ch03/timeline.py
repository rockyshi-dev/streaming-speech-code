"""One utterance through the CPU-track streaming Zipformer (sherpa-onnx, 1 thread, 160 ms client
chunks, model chunk 320 ms), at real-time pace: words on screen after each chunk, the chunks that
triggered encoder work, and the moments chapter 3 names. Data: data/timeline.json, exported from
the harness log of code/ch03_latency/harness_demo.py."""
import json
import pathlib

from figures.bookstyle import setup, figure, save

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "timeline.json"))
ch = d["chunks"]
on, off = d["speech_vad_s"]
k = 1000.0  # seconds -> ms
fig, ax = figure("wide", aspect=0.42)
ax.axvspan(on * k, off * k, color=R.past, alpha=0.25, lw=0)
ax.text(on * k + 100, d["final_words"] + 1.2, "speech (VAD)", fontsize=9)
t = [0.0] + [c["done_s"] * k for c in ch] + [d["final_done_s"] * k]
w = [0] + [c["words"] for c in ch] + [d["final_words"]]
ax.step(t, w, where="post", color=R.focus, lw=1.6, label="words on screen")
for c in ch:
    if c["compute_ms"] > 5:
        ax.plot([c["arrival_s"] * k] * 2, [-1.6, -0.6], color=R.lat.compute, lw=1.2)
ax.text(20, -2.9, "encoder runs (every 320 ms, once about 458 ms of audio exist)", fontsize=9)
first = next(c for c in ch if c["words"] > 0)
ax.annotate(f"first word: {first['done_s'] * k:,.0f} ms\n({(first['done_s'] - on) * k:,.0f} ms after speech starts)",
            xy=(first["done_s"] * k, 1), xytext=(first["done_s"] * k + 600, 8), fontsize=9,
            arrowprops=dict(arrowstyle="->", color=C.ink, lw=0.8))
ax.annotate(f"final: {d['final_done_s'] * k:,.0f} ms\n({(d['final_done_s'] - off) * k:,.0f} ms after speech ends)",
            xy=(d["final_done_s"] * k, d["final_words"]), xytext=(d["final_done_s"] * k - 3000, d["final_words"] - 18),
            fontsize=9, arrowprops=dict(arrowstyle="->", color=C.ink, lw=0.8))
ax.set_xlim(0, (d["audio_s"] + 0.4) * k)
ax.set_ylim(-3.5, d["final_words"] + 3)
ax.set_xlabel("time since the first sample was spoken (ms)")
ax.set_ylabel("words shown")
save(fig)
