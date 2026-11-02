"""Small shared helpers for the serving and cascade harnesses (chapters 15-18).

Everything here is CPU-only and dependency-light (numpy, soundfile, scipy) so the unit tests run anywhere.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
import unicodedata
from math import gcd
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import paths

REPO = Path(__file__).resolve().parents[3]
RUN_ROOT = paths.book_data("runs")
RESULTS = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------- audio

def load_audio(path: str | Path, sr: int = 16000) -> np.ndarray:
    """Read a wav/flac file as mono float32 in [-1, 1] at `sr` Hz."""
    import soundfile as sf

    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != sr:
        x = resample(x, fs, sr)
    return x.astype(np.float32)


def resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Polyphase resampling (scipy), exact for integer rate ratios."""
    from scipy.signal import resample_poly

    g = gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


def float_to_s16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def s16_to_float(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def chunk_samples(x: np.ndarray, n: int) -> list[np.ndarray]:
    """Split x into consecutive pieces of n samples; the last piece may be shorter."""
    return [x[i : i + n] for i in range(0, len(x), n)]


# ---------------------------------------------------------------- statistics

def pct(values: Sequence[float], q: float) -> float | None:
    """Percentile with linear interpolation; None for an empty list."""
    v = [float(a) for a in values if a is not None]
    if not v:
        return None
    return float(np.percentile(np.asarray(v), q))


def summarize(values: Sequence[float], scale: float = 1.0, ndigits: int = 1) -> dict:
    """p50/p95/mean/max/count of a list, multiplied by `scale` (e.g. 1000 for s -> ms)."""
    v = [float(a) * scale for a in values if a is not None]
    if not v:
        return {"n": 0}
    arr = np.asarray(v)
    r = lambda a: round(float(a), ndigits)  # noqa: E731
    return {
        "n": len(v),
        "p50": r(np.percentile(arr, 50)),
        "p95": r(np.percentile(arr, 95)),
        "mean": r(arr.mean()),
        "max": r(arr.max()),
    }


# ---------------------------------------------------------------- scoring

_PUNCT = re.compile(r"[^\w\s']", flags=re.UNICODE)


def is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿"


def normalize_text(s: str, lang: str) -> list[str]:
    """Scoring units. en: lowercase words without punctuation; zh: CJK characters (and latin words)."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = _PUNCT.sub(" ", s)
    if lang == "zh":
        units: list[str] = []
        for tok in s.split():
            buf = ""
            for ch in tok:
                if is_cjk(ch):
                    if buf:
                        units.append(buf)
                        buf = ""
                    units.append(ch)
                else:
                    buf += ch
            if buf:
                units.append(buf)
        return units
    return s.split()


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return d[len(hyp)]


def error_rate(pairs: Iterable[tuple[str, str, str]]) -> dict:
    """pairs of (ref, hyp, lang) -> {'errors', 'units', 'rate'} pooled over all pairs."""
    e = n = 0
    for ref, hyp, lang in pairs:
        r = normalize_text(ref, lang)
        e += edit_distance(r, normalize_text(hyp, lang))
        n += len(r)
    return {"errors": e, "units": n, "rate": round(e / n, 4) if n else None}


# ---------------------------------------------------------------- run metadata

def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # pragma: no cover - best effort metadata
        return None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def gpu_name() -> str | None:
    out = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return out or None


def pip_versions(names: Iterable[str]) -> dict:
    from importlib import metadata

    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def run_meta(packages: Iterable[str] = (), extra: dict | None = None) -> dict:
    meta = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": cpu_model(),
        "cpu_threads": os.cpu_count(),
        "gpu": gpu_name(),
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "git_commit": _cmd(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]),
        "loadavg_start": os.getloadavg(),
        "packages": pip_versions(packages),
    }
    if extra:
        meta.update(extra)
    return meta


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(paths.portable_tree(obj), indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def read_utts(path: str | Path) -> list[dict]:
    """The utterance manifest (data/test_utts.jsonl): each `path` is stored relative to $BOOK_DATASETS."""
    rows = read_jsonl(path)
    for r in rows:
        r["path"] = str(paths.dataset(r["path"])) if not os.path.isabs(r["path"]) else r["path"]
    return rows
