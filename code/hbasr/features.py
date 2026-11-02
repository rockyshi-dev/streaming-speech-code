"""A log-mel front end in plain numpy, and streaming-safe vs unsafe normalization (chapter 2).

`kaldi_fbank` reproduces Kaldi's `compute-fbank-feats` (and torchaudio.compliance.kaldi.fbank) for the
options used by icefall/sherpa-onnx streaming models: 25 ms Povey window, 10 ms hop, pre-emphasis 0.97,
DC removal, 512-point FFT, power spectrum, Kaldi mel scale, natural log.
"""
from __future__ import annotations

import numpy as np

EPS = np.finfo(np.float32).eps  # Kaldi's log floor (std::numeric_limits<float>::epsilon())


def kaldi_mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def kaldi_mel_banks(n_mels: int, n_fft: int, sample_rate: int, low_freq: float, high_freq: float) -> np.ndarray:
    """Triangular filters equally spaced on the Kaldi mel scale, evaluated at FFT bin frequencies.

    Returns (n_mels, n_fft // 2 + 1); the Nyquist bin gets weight 0, as in Kaldi.
    """
    nyquist = 0.5 * sample_rate
    if high_freq <= 0:
        high_freq = nyquist + high_freq
    n_bins = n_fft // 2
    bin_hz = sample_rate / n_fft
    mel_lo, mel_hi = kaldi_mel(low_freq), kaldi_mel(high_freq)
    delta = (mel_hi - mel_lo) / (n_mels + 1)
    mel_f = kaldi_mel(bin_hz * np.arange(n_bins))
    banks = np.zeros((n_mels, n_bins + 1), dtype=np.float64)
    for m in range(n_mels):
        left, center, right = mel_lo + m * delta, mel_lo + (m + 1) * delta, mel_lo + (m + 2) * delta
        up = (mel_f - left) / (center - left)
        down = (right - mel_f) / (right - center)
        banks[m, :n_bins] = np.maximum(0.0, np.minimum(up, down))
    return banks


def povey_window(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))) ** 0.85


def frame_signal(x: np.ndarray, frame_len: int, hop: int, snip_edges: bool) -> np.ndarray:
    """Cut `x` into overlapping frames the way Kaldi does.

    snip_edges=True: only frames that fit entirely, count = 1 + (N - frame_len) // hop.
    snip_edges=False: count = (N + hop // 2) // hop, frames centred on m * hop + hop / 2, reflected at edges.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if snip_edges:
        if n < frame_len:
            return np.zeros((0, frame_len))
        count = 1 + (n - frame_len) // hop
        idx = np.arange(frame_len)[None, :] + hop * np.arange(count)[:, None]
        return x[idx]
    count = (n + hop // 2) // hop
    offset = hop // 2 - frame_len // 2
    idx = np.arange(frame_len)[None, :] + hop * np.arange(count)[:, None] + offset
    # Kaldi's reflection: s < 0 -> -s - 1 ; s >= n -> 2n - 1 - s
    idx = np.where(idx < 0, -idx - 1, idx)
    idx = np.where(idx >= n, 2 * n - 1 - idx, idx)
    return x[idx]


def kaldi_fbank(
    x: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    preemph: float = 0.97,
    remove_dc: bool = True,
    snip_edges: bool = True,
) -> np.ndarray:
    """Kaldi-compatible log-mel filterbank, (frames, n_mels). Dither is always 0 here."""
    frame_len = int(sample_rate * frame_ms / 1000)
    hop = int(sample_rate * hop_ms / 1000)
    n_fft = 1 << (frame_len - 1).bit_length()
    frames = frame_signal(x, frame_len, hop, snip_edges)
    if remove_dc:
        frames = frames - frames.mean(axis=1, keepdims=True)
    if preemph:
        prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
        frames = frames - preemph * prev
    frames = frames * povey_window(frame_len)[None, :]
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    banks = kaldi_mel_banks(n_mels, n_fft, sample_rate, low_freq, high_freq)
    mel = spec @ banks.T
    return np.log(np.maximum(mel, EPS)).astype(np.float32)


# ---------------------------------------------------------------- normalization

def utterance_cmvn(feats: np.ndarray, variance: bool = True) -> np.ndarray:
    """Per-utterance mean (and variance) normalization: every frame uses statistics of ALL frames,
    including frames that have not been spoken yet. Not streaming-safe."""
    mu = feats.mean(axis=0, keepdims=True)
    out = feats - mu
    if variance:
        out = out / (feats.std(axis=0, keepdims=True) + 1e-5)
    return out


def causal_cmvn(feats: np.ndarray, variance: bool = True, prior_mean=None, prior_var=None,
                prior_count: float = 0.0) -> np.ndarray:
    """Running (cumulative) normalization: frame t uses statistics of frames 0..t only, optionally
    blended with prior statistics (e.g. global training statistics) worth `prior_count` frames."""
    feats = feats.astype(np.float64)
    d = feats.shape[1]
    pm = np.zeros(d) if prior_mean is None else np.asarray(prior_mean, dtype=np.float64)
    pv = np.ones(d) if prior_var is None else np.asarray(prior_var, dtype=np.float64)
    s1 = np.cumsum(feats, axis=0) + prior_count * pm
    s2 = np.cumsum(feats ** 2, axis=0) + prior_count * (pv + pm ** 2)
    n = np.arange(1, feats.shape[0] + 1)[:, None] + prior_count
    mu = s1 / n
    out = feats - mu
    if variance:
        var = np.maximum(s2 / n - mu ** 2, 0.0)
        out = out / (np.sqrt(var) + 1e-5)
    return out.astype(np.float32)


def sliding_cmn(feats: np.ndarray, window: int = 600) -> np.ndarray:
    """Kaldi `apply-cmvn-sliding --center=false`: subtract the mean of the last `window` frames (t-window+1..t)."""
    c = np.cumsum(np.vstack([np.zeros((1, feats.shape[1])), feats.astype(np.float64)]), axis=0)
    t = np.arange(1, feats.shape[0] + 1)
    lo = np.maximum(0, t - window)
    mu = (c[t] - c[lo]) / (t - lo)[:, None]
    return (feats - mu).astype(np.float32)


def prefix_leak(normalize, feats: np.ndarray, prefix_frames: int) -> float:
    """Max |difference| on the first `prefix_frames` frames between normalizing the prefix alone and
    normalizing the whole utterance. 0 means the normalizer never looked at the future."""
    a = normalize(feats[:prefix_frames])
    b = normalize(feats)[:prefix_frames]
    return float(np.max(np.abs(a - b)))
