import filecmp
import os

from hbasr import paths

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_copies_identical():
    a = os.path.join(CODE, "hbasr", "paths.py")
    for b in ("serve/common/paths.py", "tts/hbtts/paths.py"):
        assert filecmp.cmp(a, os.path.join(CODE, b), shallow=False), b


def test_roundtrip_roots():
    for name, root in paths.ROOTS.items():
        p = os.path.join(str(root), "runs", "x.json")
        s = paths.portable(p)
        assert s == f"${name}/runs/x.json"
        assert paths.resolve(s) == p


def test_repo_relative_and_strings():
    p = str(paths.REPO / "code" / "ch09_measure" / "eval_sherpa.py")
    assert paths.portable(p) == "code/ch09_measure/eval_sherpa.py"
    cmd = f"python {p} --out {paths.BOOK_DATA}/runs/a kokoro:{paths.BOOK_MODELS}/m"
    assert paths.portable(cmd) == "python code/ch09_measure/eval_sherpa.py --out $BOOK_DATA/runs/a kokoro:$BOOK_MODELS/m"
    assert not paths.portable(str(paths.BOOK_DATA) + "X/y").startswith("$BOOK_DATA")  # a different directory


def test_conda_and_idempotent():
    exe = "/opt/miniconda3/envs/nemo/bin/python"
    assert paths.portable(exe) == "python (conda env nemo)"
    s = paths.portable("/opt/miniconda3/envs/tts/lib/python3.12/site-packages/x.yaml")
    assert s == "<conda env tts>/lib/python3.12/site-packages/x.yaml"
    tree = {"a": [f"{paths.BOOK_DATASETS}/c", 1.5, None], "b": {"c": exe}}
    once = paths.portable_tree(tree)
    assert paths.portable_tree(once) == once
    assert once["a"][1] == 1.5


def test_long_paths_are_fast():
    # a token with many path segments that is not a conda path must not backtrack exponentially
    import time
    s = "/" + "/".join(f"seg{i}" for i in range(40)) + "/file.txt"
    t = time.perf_counter()
    assert paths.portable(s) == s
    assert time.perf_counter() - t < 0.5


def test_hf_home_is_a_root():
    p = str(paths.hf_home() / "hub" / "models--x" / "snapshots" / "abc" / "m.nemo")
    s = paths.portable(p)
    assert s == "$HF_HOME/hub/models--x/snapshots/abc/m.nemo"
    assert paths.resolve(s) == p
