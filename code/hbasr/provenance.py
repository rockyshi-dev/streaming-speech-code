"""What every result file records: command, versions, machine, git state, time.

No host name and no absolute path of the machine: paths are written through `paths.portable`.
"""
from __future__ import annotations

import datetime
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from . import paths

PACKAGES = ["numpy", "torch", "torchaudio", "sherpa-onnx", "onnxruntime", "silero-vad", "kaldi-native-fbank",
            "jiwer", "nemo_toolkit", "soundfile", "cn2an", "opencc-python-reimplemented", "sentencepiece"]


def _git(*args):
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(["git", "-C", root, *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def gpu_name():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                       text=True, stderr=subprocess.DEVNULL).strip().splitlines()
    except Exception:
        return None


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def check_same_repo() -> str:
    """Stop unless the hbasr being imported lies in the same repository as the script that is running.

    Every script puts its own repository's code/ first on sys.path; an editable install of another
    checkout would otherwise run that checkout's library code and record its git commit. Returns the
    hbasr directory as provenance records it."""
    hb = Path(paths.__file__).resolve().parent
    script = sys.argv[0] if sys.argv and sys.argv[0] not in ("", "-c") else None
    if script:
        mine = paths.repo_of(os.path.abspath(script))
        if mine != paths.REPO:
            raise SystemExit(f"hbasr is imported from {hb}, but the script {script} belongs to "
                             f"{mine or 'no book repository'}; run the script from its own repository, whose "
                             f"code/ directory it puts first on sys.path")
    return paths.portable(str(hb))


def collect(extra: dict | None = None) -> dict:
    hbasr_dir = check_same_repo()
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return {
        "when_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
        "cwd": paths.cwd(),
        "hbasr": hbasr_dir,
        "python": sys.version.split()[0],
        "packages": versions,
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "loadavg_at_start": os.getloadavg(),
        "gpu": gpu_name() if os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" else None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "code")),
    } | (extra or {})


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(paths.portable_tree(obj), f, ensure_ascii=False, indent=2, default=float)
        f.write("\n")
