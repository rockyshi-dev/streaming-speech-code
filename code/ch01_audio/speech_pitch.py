"""Chapter 1: what speech looks like. Pitch of a real recording from the autocorrelation of its samples.

    python code/ch01_audio/speech_pitch.py [--utt LANG MANIFEST UTT_ID ...] [--out FILE]

For one English and one Mandarin utterance of the book's test sets (LibriSpeech test-clean, AISHELL-1 test):
1. cut the samples into 40 ms analysis frames every 10 ms;
2. for each frame, compute the normalized autocorrelation r(lag) between the frame and itself shifted by `lag`
   samples, and take the shortest lag in 2.5-12.5 ms (400-80 Hz) whose r is within 10% of the largest r;
3. call a frame voiced when that r is at least 0.6, the frame is not more than 30 dB below the loudest frame, and
   the best lag is not at the edge of the search range (a hum or a rumble repeats at any lag; speech does not);
4. pick two 100 ms stretches by rule, without looking at the transcript: the "voiced stretch" (the loudest
   stretch in which every hop is steady and voiced) and the "noise stretch" (the loudest stretch in which no hop
   repeats and the signal crosses zero often); measure the period of the voiced stretch;
5. check the estimator on a synthetic harmonic signal of known period.
Writes results/speech_pitch.json, or --out (the default file is read by figures/ch01/data/export_speech_scales.py).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, paths, provenance  # noqa: E402

HERE = Path(__file__).resolve().parent
MANIFESTS = HERE.parent / "data" / "manifests"
DEFAULT_UTTS = [("en", "librispeech-test-clean-200", "5142-33396-0016"),
                ("zh", "aishell1-test-200", "BAC009S0904W0260")]

FRAME_MS, HOP_MS, WINDOW_MS = 40, 10, 100
F0_MIN, F0_MAX = 80.0, 400.0
VOICED_R, ENERGY_FLOOR_DB = 0.6, -30.0
STEADY_R, STEADY_TOL, NOISE_R_MAX = 0.7, 0.15, 0.5  # selection of the two 100 ms stretches, see best_window
NOISE_FLOOR_DB, NOISE_MIN_ZCR = -40.0, 4000.0  # a noise-like stretch must not be silence (energy) and must cross
                                               # zero often (noise)


def norm_autocorr(seg: np.ndarray, lag_min: int, lag_max: int) -> np.ndarray:
    """r[k] for lags lag_min..lag_max: sum x[n] x[n+lag] divided by sqrt(sum x[n]^2 * sum x[n+lag]^2).
    r is 1 when the signal repeats exactly after `lag` samples and near 0 when it does not."""
    x = seg.astype(np.float64) - seg.mean()
    out = np.zeros(lag_max - lag_min + 1)
    for i, lag in enumerate(range(lag_min, lag_max + 1)):
        a, b = x[: x.size - lag], x[lag:]
        den = np.sqrt(np.dot(a, a) * np.dot(b, b))
        out[i] = np.dot(a, b) / den if den > 0 else 0.0
    return out


def estimate_period(seg: np.ndarray, sr: int) -> tuple[float, float]:
    """(period in samples with parabolic refinement, r at the peak) for one segment.
    Takes the shortest local maximum within 10% of the largest r, which avoids picking 2 or 3 periods."""
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    r = norm_autocorr(seg, lo, hi)
    best = r.max()
    peaks = [i for i in range(1, r.size - 1) if r[i] >= r[i - 1] and r[i] > r[i + 1] and r[i] >= 0.9 * best]
    i = peaks[0] if peaks else int(r.argmax())
    frac = 0.0
    if 0 < i < r.size - 1:
        d = r[i - 1] - 2 * r[i] + r[i + 1]
        frac = 0.5 * (r[i - 1] - r[i + 1]) / d if d != 0 else 0.0
    return lo + i + frac, float(r[i])


def track(x: np.ndarray, sr: int) -> dict:
    """Per-hop pitch track: frame centres, rms (dB re the loudest frame), r peak, period (samples), voiced flag."""
    n, h = sr * FRAME_MS // 1000, sr * HOP_MS // 1000
    xp = np.pad(x, (n // 2, n // 2))
    starts = range(0, xp.size - n + 1, h)
    rms, rpk, per = [], [], []
    for s in starts:
        seg = xp[s:s + n]
        rms.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))
        p, r = estimate_period(seg, sr) if rms[-1] > 0 else (0.0, 0.0)
        per.append(p)
        rpk.append(r)
    rms = np.array(rms)
    rms_db = 20 * np.log10(np.maximum(rms, 1e-9) / rms.max())
    rpk, per = np.array(rpk), np.array(per)
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    inside = (per > lo + 1) & (per < hi - 1)
    voiced = (rpk >= VOICED_R) & (rms_db >= ENERGY_FLOOR_DB) & inside
    return {"rms": rms, "rms_db": rms_db, "r": rpk, "period": per, "voiced": voiced}


def best_window(tr: dict, x: np.ndarray, sr: int, kind: str) -> int | None:
    """Start hop of the 100 ms (10 hops) stretch with the most energy in which every hop is steady voiced
    (r >= STEADY_R, period within STEADY_TOL of the stretch median) or, for 'unvoiced', every hop with r < NOISE_R_MAX, not silent
    and a stretch with at least NOISE_MIN_ZCR zero crossings per second."""
    k = WINDOW_MS // HOP_MS
    best, best_e = None, -1.0
    for i in range(len(tr["r"]) - k + 1):
        sl = slice(i, i + k)
        if kind == "voiced":
            if not (tr["r"][sl] >= STEADY_R).all() or not tr["voiced"][sl].all():
                continue
            p = tr["period"][sl]
            if np.abs(p / np.median(p) - 1).max() > STEADY_TOL:
                continue
        else:
            if not ((tr["r"][sl] < NOISE_R_MAX).all() and (tr["rms_db"][sl] >= NOISE_FLOOR_DB).all()):
                continue
            a = i * sr * HOP_MS // 1000
            if zero_crossings_per_s(x[a:a + sr * WINDOW_MS // 1000], sr) < NOISE_MIN_ZCR:
                continue
        e = float(tr["rms"][sl].mean())
        if e > best_e:
            best, best_e = i, e
    return best


def zero_crossings_per_s(seg: np.ndarray, sr: int) -> float:
    s = np.signbit(seg - seg.mean())
    return float(np.count_nonzero(s[1:] != s[:-1]) / (seg.size / sr))


def power_fraction_above(seg: np.ndarray, sr: int, hz: float) -> float:
    """Share of the segment's power (Hann-windowed FFT) at frequencies above `hz`."""
    p = np.abs(np.fft.rfft(seg.astype(np.float64) * np.hanning(seg.size))) ** 2
    f = np.fft.rfftfreq(seg.size, 1 / sr)
    return float(p[f > hz].sum() / p.sum())


def analyze(lang: str, manifest: str, utt: str) -> dict:
    with open(MANIFESTS / f"{manifest}.tsv", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f, delimiter="\t") if r["utt_id"] == utt)
    path = paths.BOOK_DATASETS / row["path"]
    x, sr = audio.read_wav(str(path))
    tr = track(x, sr)
    hop_s = HOP_MS / 1000
    f0 = sr / tr["period"][tr["voiced"]]
    out = {"language": lang, "manifest": manifest, "utt_id": utt, "text": row["text"], "path": str(row["path"]),
           "n_words_as_transcribed": len(row["text"].split()),
           "n_characters_without_spaces": len("".join(row["text"].split())),
           "sample_rate": sr, "n_samples": int(x.size), "duration_s": x.size / sr,
           "n_hops": int(tr["voiced"].size), "n_voiced_hops": int(tr["voiced"].sum()),
           "voiced_fraction": float(tr["voiced"].mean()),
           "f0_median_hz": float(np.median(f0)), "f0_p10_hz": float(np.percentile(f0, 10)),
           "f0_p90_hz": float(np.percentile(f0, 90))}
    i = best_window(tr, x, sr, "voiced")
    n = sr * WINDOW_MS // 1000
    if i is not None:
        a = i * sr * HOP_MS // 1000  # the stretch starts at the centre of hop i's frame
        seg = x[a:a + n]
        p, r = estimate_period(seg, sr)
        out["voiced_stretch"] = {"start_ms": a / sr * 1000, "length_ms": WINDOW_MS, "period_samples": p,
                        "period_ms": p / sr * 1000, "f0_hz": sr / p, "r_peak": r,
                        "periods_in_window": WINDOW_MS / 1000 * sr / p,
                        "rms_dbfs": audio.dbfs(seg), "zero_crossings_per_s": zero_crossings_per_s(seg, sr),
                        "power_fraction_above_4khz": power_fraction_above(seg, sr, 4000.0)}
    j = best_window(tr, x, sr, "unvoiced")
    if j is not None:
        a = j * sr * HOP_MS // 1000
        seg = x[a:a + n]
        _, r = estimate_period(seg, sr)
        out["noise_stretch"] = {"start_ms": a / sr * 1000, "length_ms": WINDOW_MS, "r_peak": r,
                            "rms_dbfs": audio.dbfs(seg), "zero_crossings_per_s": zero_crossings_per_s(seg, sr),
                            "power_fraction_above_4khz": power_fraction_above(seg, sr, 4000.0)}
    return out


def selfcheck(sr: int = 16000) -> dict:
    """A harmonic signal (fundamental plus 4 overtones, amplitudes 1/k) with a known period, plus noise-free."""
    rows = []
    for f0 in (100.0, 125.0, 210.0, 330.0):
        t = np.arange(int(0.1 * sr)) / sr
        s = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))
        p, r = estimate_period(s.astype(np.float32), sr)
        rows.append({"true_f0_hz": f0, "estimated_f0_hz": sr / p, "r_peak": r})
    return {"signal": "sum of sin(2 pi k f0 t)/k for k = 1..5, 100 ms at 16 kHz", "cases": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utt", nargs=3, action="append", metavar=("LANG", "MANIFEST", "UTT_ID"),
                    help="analyze this utterance instead of the defaults (repeatable)")
    ap.add_argument("--out", default=str(HERE / "results" / "speech_pitch.json"),
                    help="result file (the default is the one the chapter and the figure read; "
                         "give another path when you try other recordings)")
    args = ap.parse_args()
    utts = [tuple(u) for u in args.utt] if args.utt else DEFAULT_UTTS
    res = {"provenance": provenance.collect({"utterances": [u[2] for u in utts]}),
           "method": {"frame_ms": FRAME_MS, "hop_ms": HOP_MS, "window_ms": WINDOW_MS, "f0_search_hz": [F0_MIN, F0_MAX],
                      "voiced_if_r_at_least": VOICED_R,
                      "voiced_stretch": {"every_hop_r_at_least": STEADY_R, "period_within_of_median": STEADY_TOL,
                                         "chosen": "highest mean frame energy"},
                      "noise_stretch": {"every_hop_r_below": NOISE_R_MAX, "energy_floor_db_re_loudest_frame": NOISE_FLOOR_DB,
                                        "min_zero_crossings_per_s": NOISE_MIN_ZCR, "chosen": "highest mean frame energy"}, "energy_floor_db_re_loudest_frame": ENERGY_FLOOR_DB},
           "selfcheck": selfcheck(),
           "utterances": [analyze(*u) for u in utts]}
    provenance.write_json(args.out, res)
    for c in res["selfcheck"]["cases"]:
        print(f"synthetic {c['true_f0_hz']:6.1f} Hz -> {c['estimated_f0_hz']:7.2f} Hz  r={c['r_peak']:.3f}")
    for u in res["utterances"]:
        print(f"{u['language']} {u['utt_id']} {u['duration_s']:.3f} s  voiced {u['voiced_fraction']:.0%}  "
              f"median f0 {u['f0_median_hz']:.1f} Hz")
        for k in ("voiced_stretch", "noise_stretch"):
            if k in u:
                print(f"   {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in u[k].items()))


if __name__ == "__main__":
    main()
