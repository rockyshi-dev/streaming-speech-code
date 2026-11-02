"""Chapter 2: per-utterance mean/variance normalization reads the future; a causal one does not.

    python code/ch02_features/normalization_leak.py

Test 1 (prefix). Normalize the first 1 s of features alone, as a streaming system that has heard
only 1 s must, and compare with the same frames after normalizing the whole utterance. A
normalizer that never looks ahead gives identical frames (difference 0).

Test 2 (future change). Keep the first 1 s of audio fixed and change only what comes after it
(the rest of the utterance, silence, or loud noise). Measure how much the features of the first 1 s
move. Under per-utterance normalization they move although their own audio did not change.

Writes results/normalization_leak.json and runs/ch02/normalization_leak.npz (for the figure).
"""
import os

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, features, paths, provenance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{paths.BOOK_DATA}/runs/ch02"
WAV = f"{paths.BOOK_DATASETS}/librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0006.flac"
os.makedirs(OUT, exist_ok=True)
x, sr = audio.read_wav(WAV)
F = lambda y: features.kaldi_fbank(y, sr, snip_edges=False, high_freq=-400.0)
feats = F(x)
PREFIX = 100  # frames = 1 s at a 10 ms hop
# test 2 compares only frames whose whole 25 ms window lies inside the fixed first second:
# frame t (snip_edges=False) covers samples [160 t - 120, 160 t + 280), so t <= 98 at 16 kHz.
SAFE = 99

norms = {
    "utterance_cmvn": lambda f: features.utterance_cmvn(f),
    "utterance_cmn_mean_only": lambda f: features.utterance_cmvn(f, variance=False),
    "causal_cmvn_running": lambda f: features.causal_cmvn(f),
    "causal_cmvn_with_prior_100_frames": lambda f: features.causal_cmvn(
        f, prior_mean=feats.mean(0), prior_var=feats.var(0), prior_count=100),
    "sliding_cmn_600": lambda f: features.sliding_cmn(f, 600),
    "none": lambda f: f,
}
res = {"provenance": provenance.collect({"input": WAV}), "prefix_frames": PREFIX,
       "test1_prefix_leak_max_abs": {}, "test2_future_change_mean_abs": {}}
for k, f in norms.items():
    res["test1_prefix_leak_max_abs"][k] = features.prefix_leak(f, feats, PREFIX)

head = x[: sr * 1]  # first second of audio, never changed
rng = np.random.default_rng(0)
futures = {
    "rest_of_utterance": x[sr:],
    "4_s_silence": np.zeros(4 * sr, np.float32),
    "4_s_loud_noise": (0.3 * rng.standard_normal(4 * sr)).astype(np.float32),
}
saved = {}
for k, f in norms.items():
    ref = f(F(np.concatenate([head, futures["rest_of_utterance"]])))[:SAFE]
    row = {}
    for name, fut in futures.items():
        cur = f(F(np.concatenate([head, fut])))[:SAFE]
        row[name] = float(np.abs(cur - ref).mean())
        saved[f"{k}|{name}"] = cur
    res["test2_future_change_mean_abs"][k] = row
res["test2_frames_compared"] = SAFE
res["note"] = ("causal_cmvn_with_prior uses statistics of this utterance as the prior only to give the "
               "running estimate a start; in deployment the prior is a fixed global statistic. Test 2 "
               "compares frames 0-98 only: frame 99 has a 25 ms window that reaches past 1 s.")
np.savez_compressed(os.path.join(OUT, "normalization_leak.npz"), **{k.replace("|", "__"): v for k, v in saved.items()})
provenance.write_json(os.path.join(HERE, "results", "normalization_leak.json"), res)
import json
print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))
