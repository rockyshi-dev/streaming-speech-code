"""The first 3 s of LibriSpeech test-clean 1089-134686-0006 three ways: waveform, linear power
spectrogram (257 bins, 31.25 Hz apart) and the 80-bin log-mel an icefall/sherpa-onnx model reads.
Data: data/triptych.npz, written by data/export_triptych.py."""
import pathlib

import numpy as np

from figures.bookstyle import setup, figure, save

C, R = setup()
d = np.load(pathlib.Path(__file__).parent / "data" / "triptych.npz")
wave, wr = d["wave"].astype(float), int(d["wave_rate"])
spec, mel = d["spec_db"].astype(float).T, d["logmel"].astype(float).T
fig, axes = figure("wide", aspect=0.62, nrows=3, sharex=True, height_ratios=[0.7, 1, 1])
t = np.arange(wave.size) / wr * 1000
axes[0].plot(t, wave, color=R.base, lw=0.5)
axes[0].set_ylabel("sample")
axes[0].set_title("waveform: 16,000 numbers per second", loc="left", fontsize=10)
ms = spec.shape[1] * 10
axes[1].imshow(spec, origin="lower", aspect="auto", cmap=C.seq, extent=[0, ms, 0, 8000],
               vmin=np.percentile(spec, 5), vmax=spec.max())
axes[1].set_ylabel("Hz")
axes[1].set_title("power spectrogram: 257 frequency bins every 10 ms", loc="left", fontsize=10)
axes[2].imshow(mel, origin="lower", aspect="auto", cmap=C.seq, extent=[0, ms, 0, 80],
               vmin=np.percentile(mel, 5), vmax=mel.max())
axes[2].set_ylabel("mel bin")
axes[2].set_title("log-mel features: 80 numbers every 10 ms (what the model reads)", loc="left", fontsize=10)
axes[2].set_xlabel("time (ms)")
for ax in axes[1:]:
    ax.grid(False)
save(fig)
