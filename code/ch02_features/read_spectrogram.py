"""Chapter 2: reading a spectrogram with numbers instead of by eye.

    python code/ch02_features/read_spectrogram.py

Two real recordings: one LibriSpeech test-clean utterance (English) and one AISHELL-1 test utterance
(Mandarin). For each, a few time ranges are fixed in `SEGMENTS` below (the names say what the ear hears
there; the ranges were chosen by looking at the spectrogram, and each is kept inside the sound it names: for every
10 ms frame of a range, `frame_checks` reports the level and the centroid).
The script then measures each range:

  silence      level in dB relative to full scale (root-mean-square, not peak), and the share of power above 4 kHz
  fricative    level, spectral centroid, share of power above 4 kHz
  stop burst   the instant of the largest 5 ms rise of the 2-8 kHz level inside a search range,
               and the size of that rise
  vowel        the pitch f0, the frequencies of the first harmonics (spectral peaks near n * f0),
               and the first three formants from a linear-prediction fit
  tone         the pitch track over the range, its start and end value, and the path of the 8th harmonic

Everything the figures draw is read from the result file, so a label always sits where the number says.
Writes results/read_spectrogram.json.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, paths, provenance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
FRAME, HOP, NFFT = 400, 160, 512  # 25 ms, 10 ms, 512-point FFT: the chapter's running example

EN_WAV = "librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0006.flac"
ZH_WAV = "aishell1/wav/test/S0764/BAC009S0764W0227.wav"

# (kind, start s, end s, what is said there).  Search ranges for bursts are wider than the burst itself.
SEGMENTS = {
    "en": {
        "wav": EN_WAV, "utt_id": "1089-134686-0006", "view": [7.0, 8.9],
        "text": "... AND TO SPREAD ABROAD ...",
        "segments": {
            "silence": ("silence", 7.15, 7.55, "pause between SLOWLY and AND"),
            "fricative": ("fricative", 7.84, 7.89, "the s of SPREAD"),
            "burst": ("burst", 7.96, 8.07, "the release of the p of SPREAD"),
            "vowel": ("vowel", 8.48, 8.66, "the vowel of ABROAD"),
        },
    },
    "zh": {
        "wav": ZH_WAV, "utt_id": "BAC009S0764W0227", "view": [0.0, 2.6],
        "text": "二零一二 保障 房 建设",
        "segments": {
            "silence": ("silence", 0.05, 0.35, "before the speaker starts"),
            "tone_fall": ("tone", 0.44, 0.62, "er (fourth tone, falling), the first syllable"),
            "tone_rise": ("tone", 0.70, 0.96, "ling (second tone, rising), the second syllable"),
            "burst_zh": ("burst", 1.54, 1.66, "the release of the zh (an affricate) of zhang"),
            "burst_j": ("burst", 2.03, 2.12, "the release of the j (an affricate) of jian"),
        },
    },
}


# ------------------------------------------------------------------ measurement helpers

def hann_frames(x: np.ndarray, frame: int = FRAME, hop: int = HOP) -> np.ndarray:
    """Frames starting at 0, hop apart, DC removed, Hann window. No pre-emphasis, so levels are true levels."""
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    f = x[idx].astype(np.float64)
    f = f - f.mean(axis=1, keepdims=True)
    return f * np.hanning(frame)[None, :]


def power_db(x: np.ndarray) -> np.ndarray:
    """(frames, 257) power spectrogram in dB, frame i covers samples 160 i .. 160 i + 400."""
    p = np.abs(np.fft.rfft(hann_frames(x), NFFT, axis=1)) ** 2
    return 10 * np.log10(np.maximum(p, 1e-12))


def cut(x: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return x[int(round(t0 * SR)): int(round(t1 * SR))]


def level_dbfs(seg: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-20))


def mean_power_spectrum(seg: np.ndarray) -> np.ndarray:
    return (np.abs(np.fft.rfft(hann_frames(seg), NFFT, axis=1)) ** 2).mean(axis=0)


def centroid_and_high_share(seg: np.ndarray) -> tuple[float, float]:
    p = mean_power_spectrum(seg)
    f = np.arange(p.size) * SR / NFFT
    return float((f * p).sum() / p.sum()), float(p[f >= 4000].sum() / p.sum())


def f0_of(frame: np.ndarray, lo: float = 70.0, hi: float = 400.0) -> tuple[float, float]:
    """Pitch by the autocorrelation peak (with a parabola through the peak), and the peak height (0 to 1)."""
    s = frame - frame.mean()
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    ac = ac / (ac[0] + 1e-20)
    a, b = int(SR / hi), int(SR / lo)
    k = a + int(np.argmax(ac[a:b]))
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return float(SR / k), float(y1)


def harmonic_peaks(seg: np.ndarray, f0: float, n: int = 8, win_ms: float = 64.0) -> list[dict]:
    """Frequency of the spectral peak nearest n * f0, from one 64 ms Hann frame in the middle of `seg`."""
    L = int(SR * win_ms / 1000)
    mid = len(seg) // 2
    fr = seg[max(0, mid - L // 2): mid + L // 2].astype(np.float64)
    fr = (fr - fr.mean()) * np.hanning(len(fr))
    nfft = 8192
    sp = np.abs(np.fft.rfft(fr, nfft))
    fq = np.arange(sp.size) * SR / nfft
    out = []
    for h in range(1, n + 1):
        band = (fq > h * f0 - 0.4 * f0) & (fq < h * f0 + 0.4 * f0)
        fp = float(fq[band][np.argmax(sp[band])])
        out.append({"n": h, "expected_hz": round(h * f0, 1), "peak_hz": round(fp, 1)})
    return out


def formants(seg: np.ndarray, order: int = 14) -> dict:
    """First three formants: median over the 25 ms frames of `seg` of the lowest three poles of a
    linear-prediction fit (autocorrelation method) with bandwidth under 500 Hz."""
    found = []
    for fr in hann_frames(np.append(seg[:1], seg[1:] - 0.97 * seg[:-1])):
        r = np.correlate(fr, fr, "full")[len(fr) - 1: len(fr) + order]
        R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)]) + 1e-9 * r[0] * np.eye(order)
        a = np.linalg.solve(R, r[1: order + 1])
        roots = np.roots(np.concatenate([[1.0], -a]))
        roots = roots[np.imag(roots) > 0]
        freq = np.angle(roots) * SR / (2 * np.pi)
        bw = -np.log(np.abs(roots)) * SR / np.pi
        ok = np.sort(freq[(freq > 150) & (freq < 4500) & (bw < 500)])
        if len(ok) >= 3:
            found.append(ok[:3])
    f = np.median(np.array(found), axis=0)
    return {"frames_used": len(found), "F1_hz": round(float(f[0]), 0), "F2_hz": round(float(f[1]), 0),
            "F3_hz": round(float(f[2]), 0)}


def burst_time(x: np.ndarray, t0: float, t1: float) -> dict:
    """The instant inside [t0, t1] where the 2-8 kHz level (10 ms Hann frames every 1 ms) rises most in 5 ms."""
    L, step = 160, 16
    hi = []
    ts = []
    for s in range(int(t0 * SR), int(t1 * SR) - L, step):
        fr = x[s: s + L].astype(np.float64)
        fr = (fr - fr.mean()) * np.hanning(L)
        sp = np.abs(np.fft.rfft(fr, 512)) ** 2
        f = np.arange(sp.size) * SR / 512
        hi.append(10 * np.log10(sp[(f >= 2000) & (f <= 8000)].sum() + 1e-12))
        ts.append((s + L / 2) / SR)
    hi, ts = np.array(hi), np.array(ts)
    k = 5  # 5 steps of 1 ms
    rise = hi[k:] - hi[:-k]
    i = int(np.argmax(rise))
    return {"burst_time_s": round(float(ts[i + k // 2 + 1]), 3), "rise_db_in_5ms": round(float(rise[i]), 1),
            "level_before_db": round(float(hi[i]), 1), "level_after_db": round(float(hi[i + k]), 1)}


def pitch_track(x: np.ndarray, t0: float, t1: float) -> dict:
    """f0 every 10 ms with a 40 ms frame; and the path of the 8th harmonic found as a spectral peak near 4 f0."""
    ts, fs, path = [], [], []
    for t in np.arange(t0, t1 - 0.02, 0.01):
        fr = x[int(round((t - 0.02) * SR)): int(round((t + 0.02) * SR))].astype(np.float64)
        f0, c = f0_of(fr)
        if c < 0.6:
            continue
        w = (fr - fr.mean()) * np.hanning(len(fr))
        sp = np.abs(np.fft.rfft(w, 8192))
        fq = np.arange(sp.size) * SR / 8192
        band = (fq > 8 * f0 - 0.4 * f0) & (fq < 8 * f0 + 0.4 * f0)
        ts.append(float(t))
        fs.append(f0)
        path.append(float(fq[band][np.argmax(sp[band])]))
    ts, fs, path = np.array(ts), np.array(fs), np.array(path)
    slope = np.polyfit(ts, fs, 1)[0] * 0.1
    return {"frames": len(ts), "f0_start_hz": round(float(fs[:3].mean()), 1), "f0_end_hz": round(float(fs[-3:].mean()), 1),
            "end_over_start": round(float(fs[-3:].mean() / fs[:3].mean()), 3),
            "slope_hz_per_100ms": round(float(slope), 1),
            "harmonic8_start_hz": round(float(path[:3].mean()), 1), "harmonic8_end_hz": round(float(path[-3:].mean()), 1),
            "harmonic8_t_start_s": round(float(ts[:3].mean()), 3), "harmonic8_t_end_s": round(float(ts[-3:].mean()), 3),
            "track_t_f0_harmonic8": [[round(float(a), 2), round(float(b), 1), round(float(c), 1)]
                                 for a, b, c in zip(ts, fs, path)]}


def frame_checks(seg: np.ndarray) -> dict:
    """Level and centroid of every 10 ms frame inside a range, so that one can see the range stays inside the sound."""
    lv, ce = [], []
    for i in range(len(seg) // 160):
        f = seg[i * 160:(i + 1) * 160].astype(np.float64)
        p = np.abs(np.fft.rfft((f - f.mean()) * np.hanning(160), NFFT)) ** 2
        fq = np.arange(p.size) * SR / NFFT
        lv.append(level_dbfs(f))
        ce.append(float((fq * p).sum() / p.sum()))
    return {"frames_10ms": len(lv), "frame_level_dbfs_min": round(min(lv), 1), "frame_level_dbfs_max": round(max(lv), 1),
            "frame_centroid_hz_min": round(min(ce), 0), "frame_centroid_hz_max": round(max(ce), 0)}


def measure(x: np.ndarray, kind: str, t0: float, t1: float) -> dict:
    seg = cut(x, t0, t1)
    out = {"kind": kind, "t_start_s": t0, "t_end_s": t1, "level_dbfs": round(level_dbfs(seg), 1)}
    if kind in ("silence", "fricative", "vowel"):
        c, h = centroid_and_high_share(seg)
        out |= {"centroid_hz": round(c, 0), "share_power_above_4khz": round(h, 3),
                "share_power_below_1khz": round(float(mean_power_spectrum(seg)[: int(np.ceil(1000 * NFFT / SR))].sum()
                                                      / mean_power_spectrum(seg).sum()), 3)} | frame_checks(seg)
    if kind == "burst":
        out |= burst_time(x, t0, t1)
    if kind == "vowel":
        fr = seg[len(seg) // 2 - 320: len(seg) // 2 + 320]
        f0, c = f0_of(fr)
        out |= {"f0_hz": round(f0, 1), "harmonics": harmonic_peaks(seg, f0), "formants": formants(seg),
                "method": "f0: autocorrelation of 40 ms; harmonic peaks: one 64 ms Hann frame, 8,192-point FFT; "
                          "formants: 14th-order linear prediction, median over the 25 ms frames of the range"}
    if kind == "tone":
        out |= pitch_track(x, t0, t1)
    return out


def analyse() -> dict:
    res = {}
    for lang, spec in SEGMENTS.items():
        x, sr = audio.read_wav(f"{paths.BOOK_DATASETS}/{spec['wav']}")
        assert sr == SR
        res[lang] = {"utt_id": spec["utt_id"], "input": spec["wav"], "text": spec["text"], "view_s": spec["view"],
                     "segments": {name: measure(x, k, a, b) | {"what": what}
                                  for name, (k, a, b, what) in spec["segments"].items()}}
    return res


if __name__ == "__main__":
    import json

    res = analyse()
    res = {"provenance": provenance.collect({"inputs": [s["wav"] for s in SEGMENTS.values()]})} | res
    provenance.write_json(os.path.join(HERE, "results", "read_spectrogram.json"), res)
    def short(x):  # leave out the frame-by-frame lists, which are only there for checking a range
        if isinstance(x, dict):
            return {k: short(v) for k, v in x.items() if not k.startswith("frame_") and k != "provenance"}
        return x

    print(json.dumps(short(res), ensure_ascii=False, indent=1))
