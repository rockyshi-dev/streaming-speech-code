"""Audio as numbers: unit arithmetic, PCM byte layout, resampling and chunking (chapter 1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------- unit arithmetic

def samples(seconds: float, sample_rate: int) -> int:
    """Number of samples in `seconds` of audio. Rounds to the nearest sample."""
    return int(round(seconds * sample_rate))


def pcm_bytes(seconds: float, sample_rate: int, bits: int = 16, channels: int = 1) -> int:
    """Bytes of uncompressed PCM for `seconds` of audio."""
    if bits % 8:
        raise ValueError("bits must be a multiple of 8")
    return samples(seconds, sample_rate) * (bits // 8) * channels


def pcm_bitrate_kbps(sample_rate: int, bits: int = 16, channels: int = 1) -> float:
    """Bitrate of uncompressed PCM in kbit/s (1 kbit = 1,000 bit)."""
    return sample_rate * bits * channels / 1000.0


def frames_in(seconds: float, frame_ms: float) -> float:
    """How many frames of `frame_ms` fit in `seconds` (not rounded)."""
    return seconds * 1000.0 / frame_ms


def dbfs(x: np.ndarray) -> float:
    """RMS level in dB relative to full scale, for float audio in [-1, 1].

    A full-scale sine has RMS 1/sqrt(2), i.e. -3.01 dBFS under this (RMS, not peak) convention.
    """
    rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    return -np.inf if rms == 0 else 20.0 * np.log10(rms)


def gain_db(x: np.ndarray, db: float) -> np.ndarray:
    """Scale amplitude by `db` decibels (-6.02 dB halves the amplitude)."""
    return (x * 10.0 ** (db / 20.0)).astype(x.dtype)


# ---------------------------------------------------------------- PCM layout

def float_to_int16(x: np.ndarray) -> np.ndarray:
    """float32 in [-1, 1] -> int16, clipping, scaling by 32767 (the convention soundfile uses for writing)."""
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def int16_to_float(x: np.ndarray) -> np.ndarray:
    """int16 -> float32 in [-1, 1), dividing by 32768 (the convention soundfile and torchaudio use for reading)."""
    return (x.astype(np.float32) / 32768.0).astype(np.float32)


def to_le_bytes(x16: np.ndarray) -> bytes:
    """int16 samples -> little-endian bytes (the WAV and most network PCM layout)."""
    return x16.astype("<i2").tobytes()


def from_bytes(buf: bytes, dtype: str) -> np.ndarray:
    """Read raw bytes as an array of `dtype` (e.g. '<i2', '>i2', '<f4'). This is where layout bugs happen."""
    itemsize = np.dtype(dtype).itemsize
    usable = len(buf) - len(buf) % itemsize
    return np.frombuffer(buf[:usable], dtype=dtype)


def interleave(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Two mono channels -> one interleaved stereo stream L0 R0 L1 R1 ..."""
    out = np.empty(left.size + right.size, dtype=left.dtype)
    out[0::2] = left
    out[1::2] = right
    return out


@dataclass
class Misread:
    name: str
    n_samples: int
    duration_s: float
    peak: float
    note: str


def misread_table(x: np.ndarray, sample_rate: int) -> list[Misread]:
    """What happens when the same 16-bit little-endian PCM bytes are read with the wrong assumptions."""
    x16 = float_to_int16(x)
    buf = to_le_bytes(x16)
    rows = []

    def add(name, y, sr, note):
        with np.errstate(invalid="ignore", over="ignore"):
            yf = y.astype(np.float64)
        finite = yf[np.isfinite(yf)]
        peak = float(np.max(np.abs(finite))) if finite.size else float("nan")
        rows.append(Misread(name, int(y.size), y.size / sr, peak, note))

    add("int16 LE (correct)", int16_to_float(from_bytes(buf, "<i2")), sample_rate, "reference")
    add("int16 BE (wrong endianness)", int16_to_float(from_bytes(buf, ">i2")), sample_rate,
        "same length, every sample's bytes swapped: loud broadband noise")
    add("float32 LE (wrong dtype)", from_bytes(buf, "<f4"), sample_rate,
        "half as many samples, values are not in [-1, 1]: garbage or NaN/inf")
    add("int16 read as stereo", int16_to_float(from_bytes(buf, "<i2"))[0::2], sample_rate,
        "every second sample dropped: half the duration if played at the original rate")
    add("16 kHz played as 8 kHz", int16_to_float(from_bytes(buf, "<i2")), sample_rate // 2,
        "same samples, wrong rate header: twice as long, one octave lower")
    return rows


# ---------------------------------------------------------------- resampling

def resample(x: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    """Band-limited resampling (torchaudio's windowed-sinc, default settings)."""
    import torch
    import torchaudio.functional as F

    if orig_sr == new_sr:
        return x.astype(np.float32)
    y = F.resample(torch.from_numpy(x.astype(np.float32)), orig_sr, new_sr)
    return y.numpy()


def naive_decimate(x: np.ndarray, factor: int) -> np.ndarray:
    """Keep every `factor`-th sample with no low-pass filter: aliases everything above the new Nyquist."""
    return x[::factor].astype(np.float32)


def tone(freq_hz: float, seconds: float, sample_rate: int, amp: float = 0.5) -> np.ndarray:
    t = np.arange(samples(seconds, sample_rate)) / sample_rate
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def band_energy_db(x: np.ndarray, sample_rate: int, lo_hz: float, hi_hz: float) -> float:
    """Energy in [lo_hz, hi_hz) relative to total energy, in dB (from one FFT of the whole signal)."""
    spec = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / sample_rate)
    band = spec[(freqs >= lo_hz) & (freqs < hi_hz)].sum()
    return 10.0 * np.log10(max(band, 1e-30) / max(spec.sum(), 1e-30))


# ---------------------------------------------------------------- chunking

@dataclass
class Chunk:
    index: int
    start_sample: int
    samples: np.ndarray

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.samples.size


def chunks(x: np.ndarray, sample_rate: int, chunk_ms: float = 160.0, pad_last: bool = False) -> Iterator[Chunk]:
    """Cut audio into consecutive chunks of `chunk_ms`.

    The last chunk is shorter unless `pad_last`, in which case it is zero-padded to full length.
    160 ms at 16 kHz is 2,560 samples.
    """
    n = samples(chunk_ms / 1000.0, sample_rate)
    if n <= 0:
        raise ValueError("chunk too small")
    for i, start in enumerate(range(0, x.size, n)):
        c = x[start:start + n]
        if pad_last and c.size < n:
            c = np.concatenate([c, np.zeros(n - c.size, dtype=x.dtype)])
        yield Chunk(i, start, c)


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV/FLAC file as float32 mono in [-1, 1)."""
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1).astype(np.float32), int(sr)
