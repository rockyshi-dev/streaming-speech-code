"""Silero VAD wrapper (chapter 7; also gives speech onset/offset references for the latency harness).

Uses the ONNX model bundled inside the `silero-vad` pip package (MIT), so nothing is downloaded.
"""
from __future__ import annotations

import numpy as np

_model = None


def model():
    global _model
    if _model is None:
        from silero_vad import load_silero_vad

        _model = load_silero_vad(onnx=True)
    return _model


def speech_segments(x: np.ndarray, sr: int = 16000, threshold: float = 0.5, min_silence_ms: int = 100,
                    min_speech_ms: int = 250, speech_pad_ms: int = 0) -> list[tuple[float, float]]:
    """Speech regions in seconds, from Silero's offline `get_speech_timestamps`."""
    import torch
    from silero_vad import get_speech_timestamps

    ts = get_speech_timestamps(torch.from_numpy(x.astype(np.float32)), model(), sampling_rate=sr,
                               threshold=threshold, min_silence_duration_ms=min_silence_ms,
                               min_speech_duration_ms=min_speech_ms, speech_pad_ms=speech_pad_ms,
                               return_seconds=False)
    return [(t["start"] / sr, t["end"] / sr) for t in ts]


def onset_offset(x: np.ndarray, sr: int = 16000) -> tuple[float, float] | None:
    segs = speech_segments(x, sr)
    if not segs:
        return None
    return segs[0][0], segs[-1][1]


class StreamingVAD:
    """Silero run causally on 32 ms windows (512 samples at 16 kHz), with a trailing-silence rule:
    speech ends when the probability stays below `threshold - 0.15` for `min_silence_ms` (the
    hysteresis Silero's VADIterator uses)."""

    window = 512

    def __init__(self, sr: int = 16000, threshold: float = 0.5, min_silence_ms: int = 500):
        from silero_vad import VADIterator

        self.it = VADIterator(model(), threshold=threshold, sampling_rate=sr,
                              min_silence_duration_ms=min_silence_ms, speech_pad_ms=0)
        self.sr = sr

    def run(self, x: np.ndarray) -> list[dict]:
        """Feed the whole signal window by window; return events with the time (s) at which each
        decision became available (= end of the window that triggered it) and the time it refers to."""
        import torch

        self.it.reset_states()
        events = []
        for a in range(0, x.size - self.window + 1, self.window):
            ev = self.it(torch.from_numpy(x[a:a + self.window].astype(np.float32)), return_seconds=False)
            if ev:
                kind = "start" if "start" in ev else "end"
                events.append({"kind": kind, "refers_to_s": ev[kind] / self.sr,
                               "decided_at_s": (a + self.window) / self.sr})
        return events
