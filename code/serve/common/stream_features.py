"""Incremental log-mel features that equal the offline features frame for frame.

An offline front end with a centred STFT (hop H samples, analysis window extending W samples either side of the
frame centre, zero padding at both ends) gives frame t centred on sample t*H. Frame t can therefore be computed
as soon as samples up to t*H + W have arrived. This class keeps the audio of one stream, and each time new
samples arrive it recomputes only a short segment that starts two hops before the first missing frame. Two hops
(2*H samples) of left margin cover the window's left half (W <= 2*H) and the one-sample memory of pre-emphasis,
so the returned frames are the offline frames, not an approximation; tests/test_stream_features.py checks it.

`mel_fn(segment) -> array (n_frames, D)` must be the offline front end applied to a segment that starts at a
multiple of H (frame k of the segment is then global frame start/H + k), returning every STFT frame it
computed (also ones the front end would mask as padding; they are never used).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class StreamFeaturizer:
    def __init__(self, mel_fn: Callable[[np.ndarray], np.ndarray], hop: int = 160, half_window: int = 256,
                 keep_s: float | None = None, sr: int = 16000, count_last_frame: bool = True):
        if half_window > 2 * hop:
            raise ValueError("half_window must be <= 2*hop for a two-hop left margin")
        self.mel_fn = mel_fn
        self.hop = hop
        self.half = half_window
        self.audio = np.zeros(0, dtype=np.float32)
        self.offset = 0          # global index of self.audio[0]
        self.n_samples = 0       # samples received so far
        self.n_frames = 0        # frames emitted so far
        self.keep = None if keep_s is None else int(keep_s * sr)
        self.finished = False
        # NeMo's front end computes len//hop + 1 frames but counts only len//hop as valid (the last, mostly
        # padding, frame is masked); count_last_frame=False reproduces that at the end of the stream.
        self.count_last = count_last_frame

    def frames_available(self) -> int:
        if self.finished:
            return self.n_samples // self.hop + (1 if self.count_last else 0)
        if self.n_samples < self.half:
            return 0
        return (self.n_samples - self.half) // self.hop + 1

    def accept(self, samples: np.ndarray) -> np.ndarray:
        """Add samples; return the newly complete frames as (n, D) (n may be 0)."""
        assert not self.finished, "accept() after finish()"
        self.audio = np.concatenate([self.audio, samples.astype(np.float32)])
        self.n_samples += len(samples)
        return self._emit()

    def finish(self) -> np.ndarray:
        """End of stream: return the remaining frames, zero-padded at the end exactly like the offline STFT."""
        self.finished = True
        return self._emit()

    def _emit(self) -> np.ndarray:
        t_end = self.frames_available()
        if t_end <= self.n_frames:
            return np.zeros((0, 0), dtype=np.float32)
        first = max(0, self.n_frames - 2)
        s0 = first * self.hop
        seg = self.audio[s0 - self.offset :]
        feats = self.mel_fn(seg)
        k0 = self.n_frames - first
        out = feats[k0 : k0 + (t_end - self.n_frames)]
        if len(out) != t_end - self.n_frames:
            raise RuntimeError(f"mel_fn returned {len(feats)} frames, expected at least {k0 + t_end - self.n_frames}")
        self.n_frames = t_end
        # drop audio that no future frame can need
        keep_from = max(0, (self.n_frames - 2) * self.hop)
        if keep_from > self.offset:
            self.audio = self.audio[keep_from - self.offset :]
            self.offset = keep_from
        return out


def reference_mel(x: np.ndarray, hop: int = 160, n_fft: int = 512, win: int = 400, n_mels: int = 16,
                  preemph: float = 0.97) -> np.ndarray:
    """A small numpy stand-in for a NeMo-style front end (pre-emphasis, centred zero-padded STFT, mel, log),
    used by the unit tests. Returns (frames, n_mels) with frames = len(x)//hop + 1."""
    x = x.astype(np.float64)
    if preemph:
        x = np.concatenate([x[:1], x[1:] - preemph * x[:-1]])
    pad = n_fft // 2
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    w = np.zeros(n_fft)
    w[(n_fft - win) // 2 : (n_fft - win) // 2 + win] = np.hanning(win)
    n = len(x) // hop + 1
    frames = np.stack([xp[t * hop : t * hop + n_fft] * w for t in range(n)])
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    fb = np.linspace(0, 1, spec.shape[1])[None, :] ** np.arange(1, n_mels + 1)[:, None]  # any fixed bank
    return np.log(spec @ fb.T + 2 ** -24).astype(np.float32)
