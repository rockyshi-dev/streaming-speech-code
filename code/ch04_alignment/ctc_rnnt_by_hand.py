"""Chapter 4: the CTC and transducer losses on examples small enough to do by hand.

    python code/ch04_alignment/ctc_rnnt_by_hand.py

For each example the script
  1. lists every alignment that produces the label sequence and its probability (brute force),
  2. computes the same total with the forward algorithm (hbasr.alignment),
  3. checks both against PyTorch's torch.nn.functional.ctc_loss / torchaudio.functional.rnnt_loss.
Probabilities are chosen as round numbers so the products can be checked with a pencil.
Writes results/ctc_rnnt_by_hand.json.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as Fnn
import torchaudio.functional as Fa

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import alignment as A  # noqa: E402
from hbasr import provenance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SYM = {0: "-", 1: "a", 2: "b"}
res = {"provenance": provenance.collect(), "examples": []}


def show(path):
    return "".join(SYM[s] for s in path)


def ctc_example(name, probs, labels):
    lp = np.log(np.asarray(probs, dtype=np.float64))
    bf, paths = A.ctc_brute_force(lp, labels)
    fw, alpha = A.ctc_forward(lp, labels, return_alpha=True)
    t = Fnn.ctc_loss(torch.tensor(lp, dtype=torch.float64)[:, None, :], torch.tensor([labels]),
                     torch.tensor([lp.shape[0]]), torch.tensor([len(labels)]), blank=0, reduction="sum").item()
    ex = {
        "loss": "CTC", "name": name, "labels": show(labels), "T": lp.shape[0],
        "probs_per_frame": probs,
        "alignments": [{"path": show(p), "prob": math.exp(l)} for p, l in paths],
        "P_total_brute_force": math.exp(-bf), "P_total_forward": math.exp(-fw),
        "forward_alpha_probs": np.exp(alpha).round(6).tolist(),
        "neg_log_P": {"brute_force": bf, "forward": fw, "torch_ctc_loss": t},
        "agree": bool(abs(bf - fw) < 1e-9 and abs(fw - t) < 1e-6),
    }
    res["examples"].append(ex)
    return ex


def rnnt_example(name, probs, labels):
    lp = np.log(np.asarray(probs, dtype=np.float64))  # (T, U+1, V)
    bf, paths = A.rnnt_brute_force(lp, labels)
    fw, alpha = A.rnnt_forward(lp, labels, return_alpha=True)
    T, U1, V = lp.shape
    t = Fa.rnnt_loss(torch.tensor(lp, dtype=torch.float32)[None], torch.tensor([labels], dtype=torch.int32),
                     torch.tensor([T], dtype=torch.int32), torch.tensor([U1 - 1], dtype=torch.int32),
                     blank=0, reduction="sum").item()
    ex = {
        "loss": "transducer", "name": name, "labels": show(labels), "T": T, "U": U1 - 1,
        "probs_t_u_v": probs,
        "alignments": [{"path": " ".join(f"({tt},{uu}):{SYM[s]}" for tt, uu, s in p), "prob": math.exp(l)}
                       for p, l in paths],
        "n_alignments": len(paths), "n_alignments_formula_C(T+U-1,U)": math.comb(T + U1 - 2, U1 - 1),
        "P_total_brute_force": math.exp(-bf), "P_total_forward": math.exp(-fw),
        "forward_alpha_probs": np.exp(alpha).round(6).tolist(),
        "neg_log_P": {"brute_force": bf, "forward": fw, "torchaudio_rnnt_loss_float32": t},
        "agree": bool(abs(bf - fw) < 1e-9 and abs(fw - t) < 1e-4),
    }
    res["examples"].append(ex)
    return ex


# CTC 1: three frames, target "ab". Five alignments: aab abb -ab a-b ab-
ctc_example("ab over 3 frames", [[0.2, 0.6, 0.2],
                                 [0.4, 0.3, 0.3],
                                 [0.1, 0.2, 0.7]], [1, 2])
# CTC 2: a repeated label needs a blank between the copies: "aa" over 3 frames has one alignment a-a
ctc_example("aa over 3 frames", [[0.2, 0.6, 0.2],
                                 [0.4, 0.3, 0.3],
                                 [0.1, 0.2, 0.7]], [1, 1])
# CTC 3: a bigger random case, no brute-force shortcut needed to trust the forward algorithm
rng = np.random.default_rng(0)
p = np.exp(A.log_softmax(rng.standard_normal((6, 3))))
ctc_example("aba over 6 frames (random)", p.round(6).tolist(), [1, 2, 1])

# transducer 1: T = 2 frames, target "a": two alignments, a on frame 0 or on frame 1
rnnt_example("a over 2 frames", [
    [[0.4, 0.6], [0.8, 0.2]],   # t=0: (u=0) P(-)=0.4 P(a)=0.6 ; (u=1) P(-)=0.8
    [[0.3, 0.7], [0.9, 0.1]],   # t=1: (u=0) P(-)=0.3 P(a)=0.7 ; (u=1) P(-)=0.9
], [1])
# transducer 2: T = 3 frames, target "ab": C(4, 2) = 6 alignments
probs = np.exp(A.log_softmax(rng.standard_normal((3, 3, 3)))).round(6)
probs = probs / probs.sum(-1, keepdims=True)
rnnt_example("ab over 3 frames (random)", probs.tolist(), [1, 2])

provenance.write_json(os.path.join(HERE, "results", "ctc_rnnt_by_hand.json"), res)
for ex in res["examples"]:
    print(f"{ex['loss']:10s} {ex['name']:28s} alignments={len(ex['alignments']):2d} "
          f"P={ex['P_total_forward']:.6f} -logP={json.dumps(ex['neg_log_P'])} agree={ex['agree']}")
for a in res["examples"][0]["alignments"]:
    print("   ", a["path"], round(a["prob"], 6))
