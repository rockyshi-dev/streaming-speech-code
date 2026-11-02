"""WER of NVIDIA's cache-aware FastConformer (stt_en_fastconformer_hybrid_large_streaming_multi) against
its worst-case look-ahead (right context x 80 ms), transducer and CTC heads: measured by the book on a
200-utterance LibriSpeech test-other subset on one RTX 4090, and the model card's numbers on the full
test-other set. Data: data/wer_vs_lookahead.json."""
import json
import pathlib

from figures.bookstyle import setup, figure, save

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "wer_vs_lookahead.json"))
fig, ax = figure(aspect=0.5)
for k, head in enumerate(["rnnt", "ctc"]):
    col = C.series[k]
    pts = sorted((r["att_context_size"][1] * 80, r["error_rate_pct"]) for r in d["measured"]
                 if r["decoder"] == head and r["manifest"].startswith("librispeech-test-other"))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col,
            label=f"{'transducer' if head == 'rnnt' else 'CTC'}, measured (200 utterances)")
    card = sorted((int(r) * 80, w) for r, w in d["card_test_other_full_set"][head].items())
    ax.plot([p[0] for p in card], [p[1] for p in card], "s--", color=col, mfc=C.paper,
            label=f"{'transducer' if head == 'rnnt' else 'CTC'}, model card (full set)")
ax.set_xlabel("worst-case look-ahead (ms)")
ax.set_ylabel("WER, LibriSpeech test-other (%)")
ax.set_xticks([0, 80, 480, 1040])
ax.legend(fontsize=8, frameon=False)
save(fig)
