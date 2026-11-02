"""Word and character error rates, with the normalization used in chapter 14.

WER = (substitutions + deletions + insertions) / number of reference words.
CER is the same count over characters (Mandarin: one CJK character = one unit).

English text is normalized by `normalize_en`: if the caller passes Whisper's
EnglishTextNormalizer (as Seed-TTS-eval does), that is used; otherwise lowercase,
drop punctuation except in-word apostrophes, collapse whitespace.
Mandarin text is normalized by `normalize_zh`: keep CJK characters, Latin letters and
digits, drop punctuation and spaces, lowercase Latin letters.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

_CJK = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"


def edit_ops(ref: Sequence, hyp: Sequence) -> tuple[int, int, int]:
    """Minimum (substitutions, deletions, insertions) turning ref into hyp.

    Ties are broken toward fewer total edits, then toward substitutions; the total is
    always the Levenshtein distance.
    """
    n, m = len(ref), len(hyp)
    # each cell: (total, subs, dels, ins)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m  # type: ignore[list-item]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cand = [prev[j - 1]]
            else:
                t, s, d, k = prev[j - 1]
                cand = [(t + 1, s + 1, d, k)]
            t, s, d, k = prev[j]
            cand.append((t + 1, s, d + 1, k))
            t, s, d, k = cur[j - 1]  # type: ignore[misc]
            cand.append((t + 1, s, d, k + 1))
            cur[j] = min(cand, key=lambda c: (c[0], -c[1]))
        prev = cur  # type: ignore[assignment]
    _, s, d, k = prev[m]
    return s, d, k


def normalize_en(text: str, whisper_normalizer: Callable[[str], str] | None = None) -> str:
    if whisper_normalizer is not None:
        return " ".join(whisper_normalizer(text).split())
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", t)
    return " ".join(t.split())


def normalize_zh(text: str) -> str:
    return "".join(re.findall(f"[{_CJK}A-Za-z0-9]", text)).lower()


def wer(ref: str, hyp: str, whisper_normalizer=None) -> dict:
    r = normalize_en(ref, whisper_normalizer).split()
    h = normalize_en(hyp, whisper_normalizer).split()
    s, d, i = edit_ops(r, h)
    return {"n_ref": len(r), "sub": s, "del": d, "ins": i,
            "rate": (s + d + i) / len(r) if r else float("nan")}


def cer(ref: str, hyp: str) -> dict:
    r = list(normalize_zh(ref))
    h = list(normalize_zh(hyp))
    s, d, i = edit_ops(r, h)
    return {"n_ref": len(r), "sub": s, "del": d, "ins": i,
            "rate": (s + d + i) / len(r) if r else float("nan")}


def pooled(rows: list[dict]) -> float:
    """Corpus-level rate: total edits / total reference units (not a mean of rates)."""
    n = sum(r["n_ref"] for r in rows)
    e = sum(r["sub"] + r["del"] + r["ins"] for r in rows)
    return e / n if n else float("nan")
