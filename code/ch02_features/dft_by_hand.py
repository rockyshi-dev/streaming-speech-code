"""Chapter 2: an 8-point discrete Fourier transform computed with nothing but a loop.

    python code/ch02_features/dft_by_hand.py

The 8 samples are integers chosen so that the table can be checked on paper. The transform is the sum written
out in the chapter, one frequency at a time, with no FFT; numpy's FFT is only used at the end to check it.
Also prints the bin spacing for the chapter's 400-sample frame. Writes results/dft_by_hand.json.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import provenance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X = [5, 6, -3, -1, -5, -6, 3, 1]
N = len(X)
SAMPLE_RATE = 16000
r = lambda v: round(float(v), 3) + 0.0  # + 0.0 turns -0.0 into 0.0


def dft_bin(x, k):
    """X[k] = sum over n of x[n] * (cos(2 pi k n / N) - i sin(2 pi k n / N)), returned as (real, imaginary)."""
    re = sum(x[n] * math.cos(2 * math.pi * k * n / N) for n in range(N))
    im = -sum(x[n] * math.sin(2 * math.pi * k * n / N) for n in range(N))
    return re, im


# the worked bin: every term of the sum for k = 1
K = 1
terms = []
for n in range(N):
    c, s = math.cos(2 * math.pi * K * n / N), math.sin(2 * math.pi * K * n / N)
    terms.append({"n": n, "x": X[n], "cos": r(c), "sin": r(s), "x_cos": r(X[n] * c), "x_sin": r(X[n] * s)})
re1, im1 = dft_bin(X, K)

bins = []
for k in range(N):
    re, im = dft_bin(X, k)
    bins.append({"k": k, "freq_hz": k * SAMPLE_RATE / N, "real": r(re), "imag": r(im), "magnitude": r(math.hypot(re, im))})
check = float(np.abs(np.array([complex(*dft_bin(X, k)) for k in range(N)]) - np.fft.fft(X)).max())

frame = {"frame_ms": 25, "frame_samples": int(SAMPLE_RATE * 0.025),
         "spacing_hz_without_padding": SAMPLE_RATE / int(SAMPLE_RATE * 0.025),
         "padded_to": 512, "spacing_hz_padded": SAMPLE_RATE / 512, "bins_padded": 512 // 2 + 1,
         "bins_unpadded": int(SAMPLE_RATE * 0.025) // 2 + 1}

# how the samples were made: a cosine of amplitude 5 that repeats once, a sine of amplitude 3 that repeats three
# times, rounded to whole numbers (the rounding is why the recovered amplitudes are not exactly 5 and 3)
assert X == [round(5 * math.cos(2 * math.pi * n / N) + 3 * math.sin(2 * math.pi * 3 * n / N)) for n in range(N)]

res = {
    "provenance": provenance.collect(),
    "built_from": "round(5 cos(2 pi n / 8) + 3 sin(2 pi 3 n / 8))",
    "samples": X, "n": N, "sample_rate_hz": SAMPLE_RATE, "duration_ms": N / SAMPLE_RATE * 1000,
    "bin_spacing_hz": SAMPLE_RATE / N,
    "worked_bin": {"k": K, "freq_hz": K * SAMPLE_RATE / N, "terms": terms,
                   "note": "totals are summed before rounding, so the rounded terms can differ from them in the last digit",
                   "real": r(re1), "imag": r(im1), "magnitude": r(math.hypot(re1, im1))},
    "bins": bins,
    "amplitude_of_cosine_at_bin": {str(b["k"]): r(2 * b["magnitude"] / N) for b in bins if 0 < b["k"] < N // 2},
    "mean_of_samples": r(sum(X) / N),
    "max_abs_difference_to_numpy_fft": check,
    "frame_400": frame,
}
provenance.write_json(os.path.join(HERE, "results", "dft_by_hand.json"), res)
import json
print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))
