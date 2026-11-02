"""Chapter 4: the two hand-worked examples of the chapter text, checked against PyTorch, and the size of
a transducer's joiner output for the CPU-track models.

    python code/ch04_alignment/chapter_examples.py

1. CTC: three frames, alphabet {a, b} plus blank, target "ab", with the per-frame probabilities of the
   chapter's table. Enumeration, the forward algorithm (hbasr.alignment) and torch.nn.functional.ctc_loss.
2. Transducer: two frames, target "ab", with the chapter's blank and next-label probabilities at each lattice
   node; the remaining probability at each node goes to the other label. Forward algorithm by hand and
   torchaudio.functional.rnnt_loss.
3. Joiner output size B x T x (U + 1) x V for one utterance per CPU-track model: V from the model's
   tokens.txt, U from its bpe.model applied to the reference transcript, T at 40 ms per encoder frame.
Writes results/chapter_examples.json.
"""
import json
import math
import os

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as Fnn
import torchaudio.functional as Fa

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import alignment as A  # noqa: E402
from hbasr import data, provenance  # noqa: E402
from hbasr import paths as bookpaths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
res = {"provenance": provenance.collect()}

# 1. CTC -----------------------------------------------------------------------------------------
P = np.array([[0.5, 0.4, 0.1], [0.3, 0.3, 0.4], [0.6, 0.1, 0.3]])  # columns: blank, a, b
paths = {"-ab": P[0, 0] * P[1, 1] * P[2, 2], "a-b": P[0, 1] * P[1, 0] * P[2, 2], "aab": P[0, 1] * P[1, 1] * P[2, 2],
         "ab-": P[0, 1] * P[1, 2] * P[2, 0], "abb": P[0, 1] * P[1, 2] * P[2, 2]}
lp = torch.log(torch.tensor(P, dtype=torch.float64))[:, None, :]
torch_loss = Fnn.ctc_loss(lp, torch.tensor([[1, 2]]), torch.tensor([3]), torch.tensor([2]), blank=0,
                          reduction="sum").item()
res["ctc"] = {"paths": {k: round(v, 6) for k, v in paths.items()}, "p_enumerated": round(sum(paths.values()), 6),
              "loss_enumerated": -math.log(sum(paths.values())), "loss_torch": torch_loss}
neg_lp = A.ctc_forward(np.log(P), [1, 2])  # returns -ln p(l | x)
res["ctc"]["loss_forward_hbasr"] = float(neg_lp)

# 2. Transducer ----------------------------------------------------------------------------------
# node (t, u): (blank prob, next-label prob); the rest goes to the other label
nodes = {(0, 0): (0.3, 0.6), (0, 1): (0.5, 0.4), (0, 2): (0.8, None),
         (1, 0): (0.2, 0.7), (1, 1): (0.2, 0.7), (1, 2): (0.9, None)}
T, U = 2, 2
probs = np.zeros((T, U + 1, 3))
for (t, u), (blank, nxt) in nodes.items():
    probs[t, u, 0] = blank
    if nxt is None:
        probs[t, u, 1] = probs[t, u, 2] = (1 - blank) / 2
    else:
        lab = 1 + u  # next label: a after u = 0, b after u = 1
        probs[t, u, lab] = nxt
        probs[t, u, 3 - lab] = 1 - blank - nxt
alpha = np.zeros((T, U + 1))
alpha[0, 0] = 1.0
for t in range(T):
    for u in range(U + 1):
        if t == 0 and u == 0:
            continue
        a = alpha[t - 1, u] * probs[t - 1, u, 0] if t > 0 else 0.0
        b = alpha[t, u - 1] * probs[t, u - 1, u] if u > 0 else 0.0
        alpha[t, u] = a + b
p_total = alpha[T - 1, U] * probs[T - 1, U, 0]
logits = torch.log(torch.tensor(probs, dtype=torch.float32))[None]
ta_loss = Fa.rnnt_loss(logits, torch.tensor([[1, 2]], dtype=torch.int32), torch.tensor([T], dtype=torch.int32),
                       torch.tensor([U], dtype=torch.int32), blank=0, reduction="sum").item()
res["transducer"] = {"alpha": alpha.round(6).tolist(), "p_forward": round(p_total, 6), "loss_forward": -math.log(p_total),
                     "loss_torchaudio_float32": ta_loss}

# 3. Joiner sizes ----------------------------------------------------------------------------------
MODELS = {"en": (f"{bookpaths.BOOK_MODELS}/sherpa-onnx-streaming-zipformer-en-2023-06-26", "librispeech-test-clean-200"),
          "zh": (f"{bookpaths.BOOK_MODELS}/sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12", "aishell1-test-200")}
res["joiner"] = {}
for lang, (mdir, manifest) in MODELS.items():
    V = sum(1 for _ in open(os.path.join(mdir, "tokens.txt"), encoding="utf-8"))
    sp = spm.SentencePieceProcessor(model_file=os.path.join(mdir, "bpe.model"))
    u = data.read_manifest(manifest)[0]
    text = u.text.replace(" ", "") if lang == "zh" else u.text
    n_tok = len(sp.encode(text))
    Tn = math.ceil(u.duration_s / 0.04)
    res["joiner"][lang] = {"utt_id": u.utt_id, "duration_s": u.duration_s, "V_tokens_txt_lines": V, "U": n_tok,
                           "T_40ms": Tn, "joiner_values": Tn * (n_tok + 1) * V,
                           "joiner_MB_fp32": Tn * (n_tok + 1) * V * 4 / 1e6}
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
provenance.write_json(os.path.join(HERE, "results", "chapter_examples.json"), res)
print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1, ensure_ascii=False))
