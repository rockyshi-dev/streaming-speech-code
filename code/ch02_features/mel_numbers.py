"""Chapter 2: numbers behind "why the mel scale and why the logarithm".

    python code/ch02_features/mel_numbers.py

Uses the repository's own front end (hbasr.features: Kaldi mel scale 1127 ln(1 + f / 700), 80 filters from
20 Hz to 7,600 Hz, 512-point FFT at 16 kHz, the options of the sherpa-onnx models) and one real utterance
(LibriSpeech test-clean 1089-134686-0006). Writes results/mel_numbers.json.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, features, paths, provenance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WAV = f"{paths.BOOK_DATASETS}/librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0006.flac"
SR, N_FFT, N_MELS, LOW, HIGH = 16000, 512, 80, 20.0, 7600.0
SHERPA = dict(snip_edges=False, high_freq=-400.0, low_freq=20.0)


def inverse_kaldi_mel(m):
    return 700.0 * (np.exp(np.asarray(m, dtype=np.float64) / 1127.0) - 1.0)


res = {"provenance": provenance.collect({"input": WAV}),
       "settings": {"sample_rate": SR, "n_fft": N_FFT, "n_mels": N_MELS, "low_hz": LOW, "high_hz": HIGH,
                    "mel_formula": "1127 * ln(1 + f / 700)"}}

# 1. the mel scale ----------------------------------------------------------------------
freqs = [500, 1000, 2000, 4000, 8000]
res["mel_of_hz"] = {str(f): round(float(features.kaldi_mel(f)), 1) for f in freqs}
d = lambda a, b: float(features.kaldi_mel(b) - features.kaldi_mel(a))
res["mel_step_of_100_hz"] = {"200_to_300_hz": round(d(200, 300), 1), "7200_to_7300_hz": round(d(7200, 7300), 1),
                             "ratio": round(d(200, 300) / d(7200, 7300), 1)}

# 2. where the 80 filters sit -------------------------------------------------------------
delta = (features.kaldi_mel(HIGH) - features.kaldi_mel(LOW)) / (N_MELS + 1)
mel_edges = features.kaldi_mel(LOW) + delta * np.arange(N_MELS + 2)
edges_hz = inverse_kaldi_mel(mel_edges)
centers = edges_hz[1:-1]
banks = features.kaldi_mel_banks(N_MELS, N_FFT, SR, LOW, -400.0)
bin_hz = SR / N_FFT
bins = np.arange(N_FFT // 2 + 1) * bin_hz
res["filters"] = {
    "center_below_1000_hz": int((centers < 1000).sum()),
    "center_1000_to_4000_hz": int(((centers >= 1000) & (centers < 4000)).sum()),
    "center_at_or_above_4000_hz": int((centers >= 4000).sum()),
    "first_filter": {"center_hz": round(float(centers[0]), 1), "width_hz": round(float(edges_hz[2] - edges_hz[0]), 1),
                     "fft_bins_with_weight": int((banks[0] > 0).sum())},
    "last_filter": {"center_hz": round(float(centers[-1]), 1), "width_hz": round(float(edges_hz[-1] - edges_hz[-3]), 1),
                    "fft_bins_with_weight": int((banks[-1] > 0).sum())},
    "fft_bins_total": int(bins.size), "fft_bin_spacing_hz": bin_hz,
    "fft_bins_below_1000_hz": int((bins < 1000).sum()),
    "fft_bins_at_or_above_4000_hz": int((bins >= 4000).sum()),
}

# 3. the logarithm on a real utterance ---------------------------------------------------------------
x, sr = audio.read_wav(WAV)
logmel = features.kaldi_fbank(x, sr, **SHERPA).astype(np.float64)
raw = np.exp(logmel)  # filter outputs before the logarithm (all above Kaldi's floor, checked below)
frame_energy = raw.sum(axis=1)
floor_hits = int((logmel <= np.log(features.EPS) + 1e-6).sum())
loud, quiet = int(frame_energy.argmax()), int(frame_energy.argmin())
res["utterance"] = {
    "utt_id": "1089-134686-0006", "frames": int(len(logmel)),
    "frame_energy_sum_of_80_filters": {
        "loudest_frame": loud, "loudest_s": round(loud * 0.01, 2), "loudest_value": float(frame_energy[loud]),
        "quietest_frame": quiet, "quietest_s": round(quiet * 0.01, 2), "quietest_value": float(frame_energy[quiet]),
        "ratio_loudest_to_quietest": float(frame_energy[loud] / frame_energy[quiet]),
        "ln_loudest": round(float(np.log(frame_energy[loud])), 3), "ln_quietest": round(float(np.log(frame_energy[quiet])), 3),
        "ln_difference": round(float(np.log(frame_energy[loud] / frame_energy[quiet])), 3)},
    "log_values_at_floor": floor_hits, "floor_value": float(features.EPS),
}
# a gain applied to the whole signal: raw values multiply, log values shift by a constant
half = features.kaldi_fbank(x * 0.5, sr, **SHERPA).astype(np.float64)
# values at Kaldi's floor do not move with the gain, so they are left out of this comparison
ok = (logmel > np.log(features.EPS) + 1e-6) & (half > np.log(features.EPS) + 1e-6)
res["gain_half"] = {"values_compared": int(ok.sum()), "values_left_out_at_floor": int((~ok).sum()),
                    "mean_change_of_log_mel": round(float((half - logmel)[ok].mean()), 4),
                    "largest_deviation_from_mean": round(float(np.abs((half - logmel)[ok] - (half - logmel)[ok].mean()).max()), 4),
                    "expected_2_ln_0.5": round(float(2 * np.log(0.5)), 4),
                    "mean_raw_ratio": round(float((np.exp(half) / raw)[ok].mean()), 4)}
provenance.write_json(os.path.join(HERE, "results", "mel_numbers.json"), res)
import json
print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))
