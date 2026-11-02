"""Chapter 2: one log-mel front end, three implementations, and how far apart they are.

    python code/ch02_features/frontend_compare.py

Computes 80-bin log-mel features of one utterance with
  (a) hbasr.features.kaldi_fbank   - the numpy front end written out in this repository,
  (b) torchaudio.compliance.kaldi.fbank,
  (c) kaldi-native-fbank           - the library sherpa-onnx itself uses at run time,
with the options sherpa-onnx's streaming Zipformer models were trained with (25 ms Povey window,
10 ms hop, 20 Hz to Nyquist - 400 Hz, no dither, snip_edges=False, samples in [-1, 1]).
Then shows what each single option change does to the features, measured as the mean absolute
difference in log-mel units. Writes results/frontend_compare.json and the feature matrices to
$BOOK_DATA/runs/ch02/frontends.npz.
"""
import os

import kaldi_native_fbank as knf
import numpy as np
import torch
import torchaudio.compliance.kaldi as K

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, features, paths, provenance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{paths.BOOK_DATA}/runs/ch02"
WAV = f"{paths.BOOK_DATASETS}/librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0006.flac"
os.makedirs(OUT, exist_ok=True)

x, sr = audio.read_wav(WAV)
SHERPA = dict(snip_edges=False, high_freq=-400.0, low_freq=20.0)

mine = features.kaldi_fbank(x, sr, **SHERPA)
ta = K.fbank(torch.from_numpy(x)[None], num_mel_bins=80, dither=0.0, sample_frequency=sr,
             snip_edges=False, high_freq=-400.0, low_freq=20.0).numpy()
o = knf.FbankOptions()
o.frame_opts.dither = 0.0
o.frame_opts.snip_edges = False
o.frame_opts.samp_freq = sr
o.mel_opts.num_bins = 80
o.mel_opts.low_freq = 20.0
o.mel_opts.high_freq = -400.0
fb = knf.OnlineFbank(o)
fb.accept_waveform(sr, x.tolist())
fb.input_finished()
kn = np.array([fb.get_frame(i) for i in range(fb.num_frames_ready)], dtype=np.float32)


def diff(a, b):
    n = min(len(a), len(b))
    d = np.abs(a[:n] - b[:n])
    return {"frames": [int(len(a)), int(len(b))], "max_abs": float(d.max()), "mean_abs": float(d.mean())}


res = {"provenance": provenance.collect({"input": WAV}), "options": SHERPA | {"n_mels": 80, "frame_ms": 25,
                                                                               "hop_ms": 10, "dither": 0}}
res["implementations"] = {
    "numpy_vs_torchaudio": diff(mine, ta),
    "numpy_vs_kaldi_native_fbank": diff(mine, kn),
    "torchaudio_vs_kaldi_native_fbank": diff(ta, kn),
}

# one option at a time -----------------------------------------------------------------
variants = {
    "snip_edges=True": dict(SHERPA, snip_edges=True),
    "high_freq=Nyquist": dict(SHERPA, high_freq=0.0),
    "no pre-emphasis": dict(SHERPA, preemph=0.0),
    "no DC removal": dict(SHERPA, remove_dc=False),
}
res["one_option_changed"] = {}
for name, kw in variants.items():
    v = features.kaldi_fbank(x, sr, **kw)
    if name == "snip_edges=True":
        # a different frame count and different frame centres, so no frame-by-frame difference
        res["one_option_changed"][name] = {"frames": [len(mine), len(v)],
                                           "note": "frame count changes: (N + 80) // 160 vs 1 + (N - 400) // 160"}
        continue
    res["one_option_changed"][name] = diff(mine, v)
# int16-scale input (Kaldi's convention when reading WAV files) vs [-1, 1] floats
v = features.kaldi_fbank(x * 32768.0, sr, **SHERPA)
d = v - mine
res["one_option_changed"]["int16-scale samples"] = diff(mine, v) | {
    "mean_offset": float(d.mean()), "expected_offset_2ln32768": float(2 * np.log(32768.0))}
# 8 kHz telephone audio upsampled to 16 kHz: the top mel bins are empty
tel = audio.resample(audio.resample(x, sr, 8000), 8000, sr)
v = features.kaldi_fbank(tel, sr, **SHERPA)
n = min(len(v), len(mine))
per_bin = np.abs(v[:n] - mine[:n]).mean(axis=0)
res["one_option_changed"]["8 kHz bandwidth audio"] = {"mean_abs_bins_0_39": float(per_bin[:40].mean()),
                                                      "mean_abs_bins_60_79": float(per_bin[60:].mean())}
np.savez_compressed(os.path.join(OUT, "frontends.npz"), numpy=mine, torchaudio=ta, knf=kn, wav=WAV)
provenance.write_json(os.path.join(HERE, "results", "frontend_compare.json"), res)
import json
print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))
