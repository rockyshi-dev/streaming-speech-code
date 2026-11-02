"""Chapter 4: CTC greedy decoding by hand on a real posterior matrix.

    python code/ch04_alignment/ctc_greedy_by_hand.py      (ssh-asr env; run export_posteriors.py first)

Takes the best token on every 80 ms frame, merges repeats, drops blanks, joins SentencePiece pieces
('▁' starts a word) and compares with NeMo's own CTC greedy transcript. Also counts how many frames
are blank, and prints the first frames' best tokens so the collapse can be followed by eye.
Writes results/ctc_greedy_by_hand.json.
"""
import os

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import alignment, paths, provenance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(f"{paths.BOOK_DATA}/runs/ch04/posteriors.npz")
lp, vocab = z["log_probs"], [str(v) for v in z["vocab"]]
blank = len(vocab) - 1
best = lp.argmax(1)
ids = alignment.ctc_greedy(lp, blank=blank)
text = "".join(vocab[i] for i in ids).replace("▁", " ").strip()
frames = [vocab[i] if i != blank else "-" for i in best]
res = {
    "provenance": provenance.collect({"input": str(z["wav"])}),
    "frames": int(lp.shape[0]), "frame_ms": int(z["frame_ms"]), "vocab_size_incl_blank": len(vocab),
    "blank_frames": int((best == blank).sum()),
    "emitted_tokens": len(ids),
    "first_40_frames_best_token": frames[:40],
    "by_hand": text, "nemo": str(z["nemo_text"]), "identical": text == str(z["nemo_text"]).strip(),
}
provenance.write_json(os.path.join(HERE, "results", "ctc_greedy_by_hand.json"), res)
print(" ".join(frames[:40]))
print("by hand:", text)
print("NeMo:   ", res["nemo"])
print("identical:", res["identical"], f"blank frames {res['blank_frames']}/{res['frames']}")
