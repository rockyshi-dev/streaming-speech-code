"""Provenance block written into every result JSON."""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from . import paths

REPO = Path(__file__).resolve().parents[3]
RUNS_ROOT = paths.book_data("runs")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "code/tts"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:
        return None


def versions(pkgs: list[str]) -> dict:
    v = {}
    for p in pkgs:
        try:
            v[p] = _md.version(p)
        except _md.PackageNotFoundError:
            v[p] = None
    return v


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def runinfo(pkgs: list[str], extra: dict | None = None) -> dict:
    info = {
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "cwd": paths.cwd(),
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "git_commit": git_commit(),
        "host_cpu": cpu_name(),
        "gpu": gpu_name(),
        "loadavg_1_5_15_at_write": list(os.getloadavg()),
        "cpu_count": os.cpu_count(),
        "packages": versions(pkgs),
    }
    if extra:
        info.update(extra)
    return info


def run_dir(name: str) -> Path:
    d = RUNS_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(paths.portable_tree(obj), indent=2, ensure_ascii=False, default=_default) + "\n")
    tmp.replace(path)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_prompts(path) -> list[dict]:
    """textset/prompts.tsv: reference clips; each `path` is stored relative to $BOOK_DATA."""
    import csv

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["path"] = str(paths.BOOK_DATA / r["path"])
    return rows


def hf_snapshot(repo: str) -> dict:
    """Resolved local snapshot (commit hash) of a HF repo in the Hugging Face cache ($HF_HOME), if present."""
    hf = paths.hf_home() / "hub"
    d = hf / ("models--" + repo.replace("/", "--"))
    ref = d / "refs" / "main"
    snaps = sorted((d / "snapshots").glob("*")) if (d / "snapshots").exists() else []
    return {"repo": repo,
            "revision": ref.read_text().strip() if ref.exists() else (snaps[-1].name if snaps else None)}
