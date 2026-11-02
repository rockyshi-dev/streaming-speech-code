"""Choose the fixed evaluation subsets once and write them as committed TSV manifests.

    python code/data/make_manifests.py

Subsets (seed 0, uniform sample without replacement from the sorted utterance list):
  librispeech-test-clean-200, librispeech-test-other-200, aishell1-test-200
"""
import os

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import data  # noqa: E402

N, SEED = 200, 0
out = data.MANIFEST_DIR
subsets = {
    "librispeech-test-clean-200": data.librispeech_all("test-clean"),
    "librispeech-test-other-200": data.librispeech_all("test-other"),
    "aishell1-test-200": data.aishell1_all("test"),
}
for name, utts in subsets.items():
    picked = data.sample(utts, N, SEED)
    data.write_manifest(picked, os.path.join(out, name + ".tsv"))
    dur = sum(u.duration_s for u in picked)
    print(f"{name}: {len(picked)} of {len(utts)} utterances, {dur / 60:.1f} min")
