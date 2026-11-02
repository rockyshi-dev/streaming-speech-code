"""Audio helpers: what counts as the first audio a listener could hear.

Definition used everywhere in chapters 10-14:

    A sample stream "becomes audible" at the first sample of the first 10 ms window
    (non-overlapping, aligned to the start of the stream) whose RMS level is at or
    above -40 dBFS. 0 dBFS is an RMS of 1.0 on float samples in [-1, 1] (int16 values
    are divided by 32768 first), so -40 dBFS is an RMS of 0.01.

Bytes of a WAV header are never audio. Leading digital silence or a quiet noise floor
before speech is audio data but not audible, so it does not end the wait for first audio.
"""

from __future__ import annotations

import struct

import numpy as np

AUDIBLE_DBFS = -40.0
AUDIBLE_WINDOW_MS = 10.0


def dbfs(rms: float) -> float:
    """RMS level in dBFS (float samples, full scale = 1.0). Silence returns -inf."""
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(rms))


def int16_bytes_to_float(buf: bytes) -> np.ndarray:
    """Little-endian signed 16-bit PCM bytes -> float32 in [-1, 1). Odd trailing byte is ignored."""
    n = len(buf) // 2
    return np.frombuffer(buf[: 2 * n], dtype="<i2").astype(np.float32) / 32768.0


def first_audible_index(
    x: np.ndarray,
    sr: int,
    threshold_dbfs: float = AUDIBLE_DBFS,
    window_ms: float = AUDIBLE_WINDOW_MS,
) -> int | None:
    """Index of the first sample of the first window whose RMS >= threshold.

    Windows are non-overlapping and start at sample 0. A final partial window counts
    if it is at least half a window long. Returns None if nothing is audible.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    win = max(1, int(round(sr * window_ms / 1000.0)))
    thr = 10.0 ** (threshold_dbfs / 20.0)
    n_full = len(x) // win
    if n_full:
        frames = x[: n_full * win].reshape(n_full, win)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        hit = np.nonzero(rms >= thr)[0]
        if len(hit):
            return int(hit[0] * win)
    tail = x[n_full * win :]
    if len(tail) >= win / 2 and np.sqrt(np.mean(tail * tail)) >= thr:
        return n_full * win
    return None


class WavStreamParser:
    """Strip a RIFF/WAVE header from a byte stream that arrives in pieces.

    Streaming servers often send a header whose size fields are 0 or 0xFFFFFFFF.
    This parser only looks for the 'data' chunk id and passes every later byte through.
    `feed` returns the PCM bytes contained in this piece (possibly empty).
    """

    def __init__(self) -> None:
        self._buf = b""
        self.header_done = False
        self.header_bytes = 0
        self.sample_rate: int | None = None
        self.channels: int | None = None
        self.bits: int | None = None

    def feed(self, piece: bytes) -> bytes:
        if self.header_done:
            return piece
        self._buf += piece
        b = self._buf
        if len(b) < 12:
            return b""
        if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
            raise ValueError("stream does not start with a RIFF/WAVE header")
        pos = 12
        while True:
            if len(b) < pos + 8:
                return b""
            cid = b[pos : pos + 4]
            csize = struct.unpack("<I", b[pos + 4 : pos + 8])[0]
            if cid == b"fmt ":
                if len(b) < pos + 8 + 16:
                    return b""
                _, ch, sr, _, _, bits = struct.unpack("<HHIIHH", b[pos + 8 : pos + 24])
                self.channels, self.sample_rate, self.bits = ch, sr, bits
            if cid == b"data":
                start = pos + 8
                self.header_done = True
                self.header_bytes = start
                rest = b[start:]
                self._buf = b""
                return rest
            if csize in (0, 0xFFFFFFFF):
                raise ValueError(f"chunk {cid!r} before 'data' has no usable size")
            pos += 8 + csize + (csize & 1)
