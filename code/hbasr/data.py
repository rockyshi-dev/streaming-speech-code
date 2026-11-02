"""Fixed, documented evaluation subsets (chapter 9).

Corpora are read in place from $BOOK_DATASETS; nothing is copied. A subset is chosen once by
`make_manifests.py` (seeded random sample of the sorted utterance list) and committed as a TSV
(`code/data/manifests/*.tsv`: utt_id, path, duration_s, text), so every run reads the same audio.
The `path` column is relative to $BOOK_DATASETS (see `paths.py`) and is resolved when a manifest is read.
"""
from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass

from . import paths

LIBRISPEECH = str(paths.dataset("librispeech", "extracted", "LibriSpeech"))
AISHELL1 = str(paths.dataset("aishell1"))
MANIFEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "manifests")


@dataclass
class Utt:
    utt_id: str
    path: str
    duration_s: float
    text: str


def librispeech_all(split: str) -> list[Utt]:
    out = []
    for trans in sorted(glob.glob(os.path.join(LIBRISPEECH, split, "*", "*", "*.trans.txt"))):
        d = os.path.dirname(trans)
        for line in open(trans, encoding="utf-8"):
            uid, text = line.rstrip("\n").split(" ", 1)
            out.append(Utt(uid, os.path.join(d, uid + ".flac"), -1.0, text))
    return sorted(out, key=lambda u: u.utt_id)


def aishell1_all(split: str = "test") -> list[Utt]:
    trans = {}
    for line in open(os.path.join(AISHELL1, "transcript", "aishell_transcript_v0.8.txt"), encoding="utf-8"):
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            trans[parts[0]] = parts[1]
    out = []
    for p in sorted(glob.glob(os.path.join(AISHELL1, "wav", split, "*", "*.wav"))):
        uid = os.path.basename(p)[:-4]
        if uid in trans:  # some AISHELL-1 wavs have no transcript; they are not part of the test set
            out.append(Utt(uid, p, -1.0, trans[uid]))
    return sorted(out, key=lambda u: u.utt_id)


def sample(utts: list[Utt], n: int, seed: int = 0) -> list[Utt]:
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(utts), size=min(n, len(utts)), replace=False).tolist())
    picked = [utts[i] for i in idx]
    for u in picked:
        u.duration_s = round(sf.info(u.path).frames / sf.info(u.path).samplerate, 4)
    return picked


def write_manifest(utts: list[Utt], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["utt_id", "path", "duration_s", "text"])
        for u in utts:
            w.writerow([u.utt_id, os.path.relpath(u.path, paths.BOOK_DATASETS), f"{u.duration_s:.4f}", u.text])


def read_manifest(name_or_path: str) -> list[Utt]:
    path = name_or_path if os.path.sep in name_or_path else os.path.join(MANIFEST_DIR, name_or_path + ".tsv")
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        return [Utt(x["utt_id"], os.path.join(paths.BOOK_DATASETS, x["path"]), float(x["duration_s"]), x["text"])
                for x in r]
