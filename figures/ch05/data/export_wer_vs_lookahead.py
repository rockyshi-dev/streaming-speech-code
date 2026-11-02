"""Export data/wer_vs_lookahead.json for wer_vs_lookahead.py from code/ch09_measure/results/summary_table.json
(the FastConformer rows; code/ch09_measure/make_tables.py writes that table), plus the model card's own
test-other figures, which the plot draws for comparison.

    python figures/ch05/data/export_wer_vs_lookahead.py [--out PATH]
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "wer_vs_lookahead.json"))
a = ap.parse_args()

table = json.load(open(REPO / "code" / "ch09_measure" / "results" / "summary_table.json"))
out = {
    "source": "code/ch09_measure/results/summary_table.json (FastConformer rows, one RTX 4090, en-lower+punct)",
    # WER (%) on the full LibriSpeech test-other set, as the model card states it (not a measurement of ours)
    "card_test_other_full_set": {
        "source": "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi README at ae98143, lines 117-125",
        "rnnt": {"13": 5.4, "6": 5.7, "1": 6.4, "0": 7.0},
        "ctc": {"13": 6.2, "6": 6.7, "1": 7.8, "0": 8.4},
    },
    "measured": [{k: r[k] for k in ("decoder", "att_context_size", "manifest", "error_rate_pct", "utterances")}
                 for r in table if r["model"] == "fastconformer-en-multi"],
}
json.dump(out, open(a.out, "w"), indent=1)
print(len(out["measured"]), "rows ->", a.out)
