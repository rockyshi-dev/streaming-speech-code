"""Word error rate against word latency (p50: when a correct word first appears for good, minus the end
of that word in a forced alignment) for every English setting the book measured on LibriSpeech
test-clean (200 utterances, 160 ms chunks, simulated real-time pacing): the CPU-track Zipformer with greedy
and beam search, and the GPU-track FastConformer at four att_context_size settings with its transducer
and CTC heads. Data: data/measured.json, from code/ch09_measure/results/summary_table.json."""
import json
import pathlib

from figures.bookstyle import setup, figure, save

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "measured.json"))
rows = [r for r in d["rows"] if r["manifest"].startswith("librispeech-test-clean") and r["word_latency_fa_p50_ms"]]
fig, ax = figure(aspect=0.5)
groups = [("fastconformer-en-multi", "rnnt", "FastConformer, transducer (GPU)", C.series[0]),
          ("fastconformer-en-multi", "ctc", "FastConformer, CTC (GPU)", C.series[1]),
          ("zipformer-en", None, "Zipformer (CPU)", R.focus)]
for model, dec, name, col in groups:
    pts = [r for r in rows if r["model"] == model and (dec is None or r["decoder"] == dec)]
    pts.sort(key=lambda r: (r["att_context_size"] or [0, 0])[1])
    ax.plot([r["word_latency_fa_p50_ms"] for r in pts], [r["error_rate_pct"] for r in pts],
            "o-" if dec else "D", color=col, label=name)
    for r in pts:
        lab = f"[70,{r['att_context_size'][1]}]" if r["att_context_size"] else ("beam" if "beam" in r["decoder"] else "greedy")
        if dec != "ctc":
            ax.annotate(lab, (r["word_latency_fa_p50_ms"], r["error_rate_pct"]), textcoords="offset points",
                        xytext=(5, -10), fontsize=8)
ax.set_xlabel("word latency, p50 (ms)")
ax.set_ylabel("WER, LibriSpeech test-clean (%)")
ax.legend(fontsize=8, frameon=False)
save(fig)
