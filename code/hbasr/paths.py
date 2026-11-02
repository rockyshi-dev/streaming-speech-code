"""Where the book's code finds corpora, models and its own run output.

Three directories, each set by an environment variable:

    BOOK_DATA     run output, prepared inputs, vendored files, caches   default <repository>/work
    BOOK_DATASETS the corpora, read in place                             default $BOOK_DATA/datasets
    BOOK_MODELS   pre-trained model directories                          default $BOOK_DATA/models

Caches (the Hugging Face cache, `HF_HOME`) default to $BOOK_DATA/cache.

Result files never store these directories as absolute paths. `portable()` writes a path under one of them
as `$BOOK_DATA/...`, `$BOOK_DATASETS/...` or `$BOOK_MODELS/...`, a path in the Hugging Face cache as
`$HF_HOME/...`, and a path inside the repository relative to
the repository root; `resolve()` turns such a string back into a path on the current machine.

The same file is used by `hbasr`, `serve/common` and `tts/hbtts` (each package imports its own copy, so each
runs in its own environment); `code/tests/test_paths.py` checks the copies are identical.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def repo_of(path) -> Path | None:
    """The repository (a directory holding code/README.md and figures/) that contains `path`, or None."""
    for p in Path(path).resolve().parents:
        if (p / "code" / "README.md").is_file() and (p / "figures").is_dir():
            return p
    return None


def _find_repo() -> Path:
    return repo_of(__file__) or Path(__file__).resolve().parents[2]


REPO = _find_repo()


def _root(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(os.path.expanduser(v)) if v else default


BOOK_DATA = _root("BOOK_DATA", REPO / "work")
BOOK_DATASETS = _root("BOOK_DATASETS", BOOK_DATA / "datasets")
BOOK_MODELS = _root("BOOK_MODELS", BOOK_DATA / "models")
ROOTS = {"BOOK_DATA": BOOK_DATA, "BOOK_DATASETS": BOOK_DATASETS, "BOOK_MODELS": BOOK_MODELS}


def book_data(*parts: str) -> Path:
    return BOOK_DATA.joinpath(*parts)


def dataset(*parts: str) -> Path:
    return BOOK_DATASETS.joinpath(*parts)


def models(*parts: str) -> Path:
    return BOOK_MODELS.joinpath(*parts)


# The Hugging Face cache: $HF_HOME, else $BOOK_DATA/cache/huggingface. It is set in the environment here so
# that huggingface_hub, transformers and NeMo, imported after this module, use the same directory.
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(BOOK_DATA / "cache" / "huggingface")


def hf_home() -> Path:
    """The Hugging Face cache directory: $HF_HOME (default $BOOK_DATA/cache/huggingface)."""
    return Path(os.environ["HF_HOME"])


# ---------------------------------------------------------------- portable strings

_PLACEHOLDER = re.compile(r"^\$(BOOK_DATA|BOOK_DATASETS|BOOK_MODELS|HF_HOME)(?=/|$)")
_CONDA = r"(?:/[^\s:=\"',/]+)*/(?:miniconda3|anaconda3|miniforge3|mambaforge|conda)/envs/([\w.+-]+)"
_CONDA_PY = re.compile(r"^" + _CONDA + r"/bin/(python[\d.]*)$")
_CONDA_PREFIX = re.compile(r"(?<![\w./-])" + _CONDA + r"(?=/)")


def _under(prefix: str) -> re.Pattern:
    # the directory itself or anything below it, as a whole path: not preceded by a path character,
    # followed by "/" or by the end of the path
    return re.compile(r"(?<![\w./-])" + re.escape(prefix.rstrip("/")) + r"(?=/|$|[^\w.-])")


def _rules(extra_roots: dict | None = None) -> list[tuple[re.Pattern, str]]:
    pairs: list[tuple[str, str]] = []
    roots = {k: [str(v)] for k, v in ROOTS.items()}
    roots["HF_HOME"] = [str(hf_home())]  # the Hugging Face cache is written as $HF_HOME/..., wherever it is
    for k, v in (extra_roots or {}).items():
        roots.setdefault(k, []).extend(v)
    for name, prefixes in roots.items():
        for p in prefixes:
            for q in {p, os.path.realpath(p)}:
                if q and q != "/":
                    pairs.append((q, "$" + name))
    for q in {str(REPO), os.path.realpath(REPO)}:
        pairs.append((q + "/", ""))
        pairs.append((q, "."))
    pairs.sort(key=lambda kv: -len(kv[0]))
    rules = []
    for prefix, repl in pairs:
        if prefix.endswith("/"):
            rules.append((re.compile(r"(?<![\w./-])" + re.escape(prefix)), repl))
        else:
            rules.append((_under(prefix), repl))
    home = os.path.expanduser("~")
    if home and home != "/":
        rules.append((_under(home), "~"))
    return rules


_RULES = _rules()


def portable(s: str, rules: list | None = None) -> str:
    """Rewrite every absolute path in `s` that lies under a root, the repository or a conda env.

    A whole-string conda Python executable becomes "python3.12 (conda env NAME)"; any other path inside a
    conda env becomes "<conda env NAME>/..."; the home directory, as a last resort, becomes "~".
    """
    if not isinstance(s, str) or "/" not in s:
        return s
    m = _CONDA_PY.match(s)
    if m:
        return f"{m.group(2)} (conda env {m.group(1)})"
    s = _CONDA_PREFIX.sub(lambda m: f"<conda env {m.group(1)}>", s)
    for pat, repl in (rules or _RULES):
        s = pat.sub(lambda _m, r=repl: r, s)
    return s


def portable_tree(obj, rules: list | None = None):
    """`portable` applied to every string (keys included) of a JSON-like object."""
    if isinstance(obj, str):
        return portable(obj, rules)
    if isinstance(obj, Path):
        return portable(str(obj), rules)
    if isinstance(obj, dict):
        return {portable_tree(k, rules) if isinstance(k, str) else k: portable_tree(v, rules)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [portable_tree(v, rules) for v in obj]
    return obj


def resolve(s: str | Path) -> str:
    """The inverse of `portable` for one path: expand a `$ROOT` prefix; a relative path is taken
    relative to the repository root unless it exists relative to the current directory."""
    s = str(s)
    m = _PLACEHOLDER.match(s)
    if m:
        root = hf_home() if m.group(1) == "HF_HOME" else ROOTS[m.group(1)]
        return str(root) + s[m.end():]
    if s.startswith("~"):
        return os.path.expanduser(s)
    if not os.path.isabs(s) and not os.path.exists(s):
        return str(REPO / s)
    return s


def cwd() -> str:
    """The working directory as provenance records it: repository-relative inside the repository."""
    return portable(os.getcwd())

