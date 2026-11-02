"""Reference word times by forced alignment (chapter 9, step 1 of the latency recipe). English only.

Aligns each LibriSpeech reference transcript to its audio with torchaudio's CTC forced aligner and the
wav2vec 2.0 Base 960 h CTC model (torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H, fairseq weights, MIT).
The aligner is a different model from every recognizer measured here, so its word times are not
biased towards any of them. Resolution: one wav2vec 2.0 frame = 20 ms.

    python -m hbasr.align_ref librispeech-test-clean-200 librispeech-test-other-200

Writes runs/align/<manifest>.json: {utt_id: [[word, start_s, end_s], ...]}.
"""
from __future__ import annotations

import json
import os
import sys

from . import audio, data, paths

RUNS = str(paths.book_data("runs"))


def path_for(manifest: str) -> str:
    return os.path.join(RUNS, "align", manifest + ".json")


def load(manifest: str):
    p = path_for(manifest)
    return json.load(open(p)) if os.path.exists(p) else None


def align_manifest(manifest: str) -> str:
    import torch
    import torchaudio
    import torchaudio.functional as F

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval()
    labels = bundle.get_labels()  # ('-', '|', 'E', 'T', ...): blank '-', word separator '|'
    idx = {c: i for i, c in enumerate(labels)}
    out = {}
    for u in data.read_manifest(manifest):
        x, sr = audio.read_wav(u.path)
        words = u.text.upper().split()
        tokens = [idx[c] for w in words for c in w if c in idx]
        with torch.inference_mode():
            em, _ = model(torch.from_numpy(x)[None])
            em = torch.log_softmax(em, dim=-1)
        ali, _ = F.forced_align(em, torch.tensor([tokens], dtype=torch.int32), blank=0)
        ali = ali[0].tolist()
        frame_s = 320 / sr  # wav2vec 2.0 feature encoder stride: 320 samples = 20 ms
        # token spans: a new token starts where the aligned label changes to a non-blank
        spans, prev = [], 0
        for t, lab in enumerate(ali):
            if lab != 0 and (lab != prev or (spans and spans[-1][2] != t - 1)):
                spans.append([lab, t, t])
            elif lab != 0 and spans:
                spans[-1][2] = t
            prev = lab
        k = 0
        row = []
        for w in words:
            n = sum(1 for c in w if c in idx)
            seg = spans[k:k + n]
            k += n
            row.append([w, seg[0][1] * frame_s, (seg[-1][2] + 1) * frame_s])
        out[u.utt_id] = row
    p = path_for(manifest)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"))
    return p


if __name__ == "__main__":
    for m in sys.argv[1:]:
        print(align_manifest(m))
