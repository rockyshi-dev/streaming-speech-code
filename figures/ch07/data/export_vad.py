"""Export the data for vad_hysteresis.py: Silero VAD (silero-vad 6.2.3) speech probability per 32 ms
window for the first 4 s of LibriSpeech test-clean 1089-134686-0006 followed by 1 s of digital silence, and the segments its
VADIterator reports with the default settings (threshold 0.5, end below 0.35 for 100 ms, 30 ms pad).
Run in the ssh-asr env from the repository root:

    PYTHONPATH=code python figures/ch07/data/export_vad.py [path/to/1089-134686-0006.flac]
"""
import json
import os
import sys

import torch
from silero_vad import VADIterator, load_silero_vad

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, paths  # noqa: E402

wav = sys.argv[1] if len(sys.argv) > 1 else (f"{paths.BOOK_DATASETS}/librispeech/extracted/LibriSpeech/test-clean/"
                                             "1089/134686/1089-134686-0006.flac")
x, sr = audio.read_wav(wav)
x = torch.cat([torch.from_numpy(x), torch.zeros(sr)])
torch.set_grad_enabled(False)
model = load_silero_vad()
probs = []
for i in range(0, x.numel() - 511, 512):
    probs.append(round(float(model(x[i:i + 512], sr)), 4))
model.reset_states()
it = VADIterator(model, sampling_rate=sr)
events = []
for i in range(0, x.numel() - 511, 512):
    e = it(x[i:i + 512], return_seconds=True)
    if e:
        events.append({k: float(v) for k, v in e.items()} | {"window_end_s": (i + 512) / sr})
out = {"source": os.path.basename(wav) + " + 1 s of silence", "window_ms": 32, "probs": probs, "events": events}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vad.json"), "w"), indent=1)
print(events)
