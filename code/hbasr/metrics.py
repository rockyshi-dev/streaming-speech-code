"""Edit distance, WER/CER and partial-result stability (chapters 6 and 9)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class EditCounts:
    ref_len: int = 0
    sub: int = 0
    dele: int = 0
    ins: int = 0

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def rate(self) -> float:
        return self.errors / self.ref_len if self.ref_len else float("nan")

    def __add__(self, o: "EditCounts") -> "EditCounts":
        return EditCounts(self.ref_len + o.ref_len, self.sub + o.sub, self.dele + o.dele, self.ins + o.ins)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["errors"] = self.errors
        d["rate"] = self.rate
        return d


def edit_counts(ref: list[str], hyp: list[str]) -> EditCounts:
    """Levenshtein alignment with unit costs; ties broken sub > del > ins (same totals as jiwer)."""
    R, H = len(ref), len(hyp)
    d = np.zeros((R + 1, H + 1), dtype=np.int64)
    d[:, 0] = np.arange(R + 1)
    d[0, :] = np.arange(H + 1)
    for i in range(1, R + 1):
        ri = ref[i - 1]
        for j in range(1, H + 1):
            c = 0 if ri == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j - 1] + c, d[i - 1, j] + 1, d[i, j - 1] + 1)
    i, j = R, H
    s = de = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            de += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return EditCounts(R, int(s), de, ins)


def corpus_rate(pairs, tokenize, normalize) -> EditCounts:
    """Sum edit counts over (ref, hyp) pairs after normalize -> tokenize; the rate is total errors over
    total reference units (not an average of per-utterance rates)."""
    total = EditCounts()
    for ref, hyp in pairs:
        total = total + edit_counts(tokenize(normalize(ref)), tokenize(normalize(hyp)))
    return total


# ---------------------------------------------------------------- stability

def common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class Stability:
    unstable_units: int
    revisions: int
    final_units: int
    partials: int


def stability(hyps: list[list[str]]) -> Stability:
    """Partial-result stability of one utterance, following Shangguan et al. 2020 (arXiv:2006.01416).

    `hyps` is the sequence of distinct partial results in emission order, the final result last.
    Between consecutive hypotheses h_i and h_{i+1}, the units of h_i after their longest common prefix
    were shown to the user and then changed: they are unstable units. A step with at least one
    unstable unit is one revision. Corpus UPWR = sum of unstable units / sum of final units;
    UPSR = sum of revisions / number of utterances.
    """
    unstable = revisions = 0
    for a, b in zip(hyps, hyps[1:]):
        k = common_prefix_len(a, b)
        changed = len(a) - k
        if changed > 0:
            unstable += changed
            revisions += 1
    return Stability(unstable, revisions, len(hyps[-1]) if hyps else 0, max(len(hyps) - 1, 0))


def upwr_upsr(per_utt: list[Stability]) -> dict:
    u = sum(s.unstable_units for s in per_utt)
    f = sum(s.final_units for s in per_utt)
    r = sum(s.revisions for s in per_utt)
    return {
        "UPWR": u / f if f else float("nan"),
        "UPSR": r / len(per_utt) if per_utt else float("nan"),
        "unstable_units": u,
        "final_units": f,
        "revisions": r,
        "utterances": len(per_utt),
    }


def percentiles(xs, ps=(50, 90, 95)) -> dict:
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=np.float64)
    if xs.size == 0:
        return {f"p{p}": None for p in ps} | {"mean": None, "n": 0}
    out = {f"p{p}": float(np.percentile(xs, p)) for p in ps}
    out["mean"] = float(xs.mean())
    out["n"] = int(xs.size)
    return out


def aligned_ref_errors(ref: list[str], hyp: list[str]) -> list[int]:
    """Indices of reference units that were substituted or deleted in the minimum-edit alignment."""
    R, H = len(ref), len(hyp)
    d = np.zeros((R + 1, H + 1), dtype=np.int64)
    d[:, 0] = np.arange(R + 1)
    d[0, :] = np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j - 1] + c, d[i - 1, j] + 1, d[i, j - 1] + 1)
    i, j, bad = R, H, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                bad.append(i - 1)
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            bad.append(i - 1)
            i -= 1
        else:
            j -= 1
    return sorted(bad)


def matched_pairs(ref: list[str], hyp: list[str]) -> list[tuple[int, int]]:
    """(ref index, hyp index) of every correctly recognized unit in the minimum-edit alignment."""
    R, H = len(ref), len(hyp)
    d = np.zeros((R + 1, H + 1), dtype=np.int64)
    d[:, 0] = np.arange(R + 1)
    d[0, :] = np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j - 1] + c, d[i - 1, j] + 1, d[i, j - 1] + 1)
    i, j, out = R, H, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] == hyp[j - 1]:
                out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]
