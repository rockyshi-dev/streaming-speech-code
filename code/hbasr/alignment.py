"""CTC and transducer (RNN-T) losses by the forward algorithm, in numpy, small enough to check by hand
(chapter 4). Everything is in natural-log space; the loss is -log P(labels | input).
"""
from __future__ import annotations

import itertools

import numpy as np


def logsumexp(*xs):
    m = max(xs)
    if m == -np.inf:
        return -np.inf
    return m + np.log(sum(np.exp(x - m) for x in xs))


# ---------------------------------------------------------------- CTC

def ctc_collapse(path, blank: int = 0) -> list[int]:
    """Merge repeated symbols, then drop blanks: (a, a, -, a, b, b) -> (a, a, b)."""
    out, prev = [], None
    for s in path:
        if s != prev and s != blank:
            out.append(int(s))
        prev = s
    return out


def ctc_greedy(log_probs: np.ndarray, blank: int = 0) -> list[int]:
    """Best symbol per frame, then collapse. `log_probs` is (T, V)."""
    return ctc_collapse(np.argmax(log_probs, axis=1).tolist(), blank)


def ctc_forward(log_probs: np.ndarray, labels, blank: int = 0, return_alpha: bool = False):
    """-log P(labels | x) summed over all CTC alignments, by the forward algorithm.

    The extended label sequence interleaves blanks: (-, l1, -, l2, ..., lU, -), length S = 2U + 1.
    alpha[t, s] = log of the total probability of all paths through frame t that end at position s.
    """
    T = log_probs.shape[0]
    ext = [blank]
    for l in labels:
        ext += [int(l), blank]
    S = len(ext)
    alpha = np.full((T, S), -np.inf)
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            terms = [alpha[t - 1, s]]
            if s >= 1:
                terms.append(alpha[t - 1, s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                terms.append(alpha[t - 1, s - 2])
            alpha[t, s] = logsumexp(*terms) + log_probs[t, ext[s]]
    end = alpha[T - 1, S - 1] if S == 1 else logsumexp(alpha[T - 1, S - 1], alpha[T - 1, S - 2])
    return (-end, alpha) if return_alpha else -end


def ctc_brute_force(log_probs: np.ndarray, labels, blank: int = 0):
    """Enumerate every length-T path, keep those that collapse to `labels`. Returns (loss, paths)."""
    T, V = log_probs.shape
    target = [int(l) for l in labels]
    kept, total = [], -np.inf
    for path in itertools.product(range(V), repeat=T):
        if ctc_collapse(path, blank) == target:
            lp = float(sum(log_probs[t, s] for t, s in enumerate(path)))
            kept.append((path, lp))
            total = logsumexp(total, lp)
    return -total, kept


# ---------------------------------------------------------------- transducer

def rnnt_forward(log_probs: np.ndarray, labels, blank: int = 0, return_alpha: bool = False):
    """-log P(labels | x) over the transducer lattice.

    `log_probs` has shape (T, U + 1, V): the joint network's log-softmax at encoder frame t after
    having emitted the first u labels. From node (t, u) a path either emits blank (moves to t + 1)
    or emits label u + 1 (moves to u + 1, same t). Every path ends with a blank from (T - 1, U).
    alpha[t, u] = log probability of reaching node (t, u).
    """
    T, U1, _ = log_probs.shape
    U = U1 - 1
    assert len(labels) == U
    alpha = np.full((T, U1), -np.inf)
    alpha[0, 0] = 0.0
    for t in range(T):
        for u in range(U1):
            if t == 0 and u == 0:
                continue
            terms = []
            if t > 0:
                terms.append(alpha[t - 1, u] + log_probs[t - 1, u, blank])
            if u > 0:
                terms.append(alpha[t, u - 1] + log_probs[t, u - 1, int(labels[u - 1])])
            alpha[t, u] = logsumexp(*terms)
    loss = -(alpha[T - 1, U] + log_probs[T - 1, U, blank])
    return (loss, alpha) if return_alpha else loss


def rnnt_brute_force(log_probs: np.ndarray, labels, blank: int = 0):
    """Enumerate every monotonic lattice path: choose where the U label emissions sit among T blanks.

    A path is a sequence of T blanks and U labels whose last symbol is a blank; there are
    C(T + U - 1, U) of them. Returns (loss, [(path, logp)]), each path as a list of (t, u, symbol).
    """
    T, U1, _ = log_probs.shape
    U = U1 - 1
    kept, total = [], -np.inf
    n = T + U - 1  # positions before the final blank
    for label_pos in itertools.combinations(range(n), U):
        t = u = 0
        lp = 0.0
        steps = []
        for i in range(n):
            if i in label_pos:
                sym = int(labels[u])
                lp += log_probs[t, u, sym]
                steps.append((t, u, sym))
                u += 1
            else:
                lp += log_probs[t, u, blank]
                steps.append((t, u, blank))
                t += 1
        lp += log_probs[T - 1, U, blank]
        steps.append((T - 1, U, blank))
        kept.append((steps, float(lp)))
        total = logsumexp(total, lp)
    return -total, kept


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))
