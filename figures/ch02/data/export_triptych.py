"""Export the data for triptych.py: the first 3 s of LibriSpeech test-clean 1089-134686-0006 as a
waveform, a linear power spectrogram in dB (25 ms Povey window, 10 ms hop, 512-point FFT) and the
80-bin log-mel of the sherpa-onnx/icefall front end (hbasr.features.kaldi_fbank, snip_edges=False,
high_freq=-400). Run in the ssh-asr env from the repository root:

    PYTHONPATH=code python figures/ch02/data/export_triptych.py [path/to/1089-134686-0006.flac]
"""
import os
import sys

import numpy as np

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, features, paths  # noqa: E402

wav = sys.argv[1] if len(sys.argv) > 1 else (f"{paths.BOOK_DATASETS}/librispeech/extracted/LibriSpeech/test-clean/"
                                             "1089/134686/1089-134686-0006.flac")
x, sr = audio.read_wav(wav)
x = x[: 3 * sr]
frames = features.frame_signal(x, 400, 160, snip_edges=False)
frames = frames - frames.mean(1, keepdims=True)
frames = np.concatenate([frames[:, :1], frames[:, 1:] - 0.97 * frames[:, :-1]], 1) * features.povey_window(400)
spec = np.abs(np.fft.rfft(frames, 512)) ** 2
spec_db = 10 * np.log10(np.maximum(spec, 1e-10))
logmel = features.kaldi_fbank(x, sr, snip_edges=False, high_freq=-400.0)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triptych.npz")
np.savez_compressed(out, wave=x[::4].astype(np.float16), wave_rate=np.array(sr // 4),
                    spec_db=spec_db.astype(np.float16), logmel=logmel.astype(np.float16))
print(out, spec_db.shape, logmel.shape)
