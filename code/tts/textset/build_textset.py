"""Build the fixed English + Mandarin text set used by chapters 12-14.

Sources (read in place from $BOOK_DATASETS, nothing is written there):
  English : LibriTTS-R test-clean, `*.normalized.txt` (CC BY 4.0, https://www.openslr.org/141/),
            read from $BOOK_DATASETS/libritts-r/test_clean.tar.gz.
  Mandarin: AISHELL-1 test transcripts (Apache-2.0, https://www.openslr.org/33/),
            $BOOK_DATASETS/aishell1.

Output (committed, small):
  textset/en.tsv, textset/zh.tsv          id <TAB> kind <TAB> text
      kind = "sent"  one utterance (20 per language), used for WER/CER
      kind = "para"  three consecutive utterances (5 per language), used for streaming
  textset/prompts.tsv                      reference clips for codec / chunking exercises (paths relative to $BOOK_DATA)
And reference wavs (not committed) under $BOOK_DATA/inputs/.

Selection is deterministic: sort by utterance id, filter by length, take evenly spaced items.
"""

from __future__ import annotations

import argparse
import collections
import re
import struct
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hbtts import paths  # noqa: E402

HERE = Path(__file__).resolve().parent
LIBRITTS_TAR = Path(f"{paths.BOOK_DATASETS}/libritts-r/test_clean.tar.gz")
AISHELL = Path(f"{paths.BOOK_DATASETS}/aishell1")
INPUTS = Path(f"{paths.BOOK_DATA}/inputs")

N_SENT, N_PARA = 20, 5


def evenly(items: list, k: int) -> list:
    if len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def load_libritts(want_prefixes: tuple[str, ...]) -> tuple[dict[str, str], dict[str, bytes]]:
    texts: dict[str, str] = {}
    wavs: dict[str, bytes] = {}
    with tarfile.open(LIBRITTS_TAR, "r:gz") as tf:
        for m in tf:
            name = Path(m.name).name
            if name.endswith(".normalized.txt"):
                uid = name[: -len(".normalized.txt")]
                texts[uid] = tf.extractfile(m).read().decode("utf-8").strip()
            elif name.endswith(".wav") and name.startswith(want_prefixes):
                wavs[name[:-4]] = tf.extractfile(m).read()
    return texts, wavs


def english(texts: dict[str, str]):
    ok = re.compile(r"^[A-Za-z0-9 ,.;:'\"?!-]+$")
    good = {u: t for u, t in texts.items()
            if ok.match(t) and t[-1] in ".?!" and 10 <= len(t.split()) <= 30 and '"' not in t and '--' not in t}
    sents = evenly(sorted(good), N_SENT)
    # paragraphs: same speaker_chapter_paragraph, >= 3 sentences, all present in `texts`
    groups = collections.defaultdict(list)
    for u in texts:
        spk, chap, para, sent = u.split("_")
        groups[(spk, chap, para)].append(u)
    paras = []
    for key in sorted(groups):
        us = sorted(groups[key])
        if len(us) < 3:
            continue
        ts = [texts[u] for u in us[:3]]
        if all(ok.match(t) and '"' not in t for t in ts) and all(t[-1] in ".?!" for t in ts):
            joined = " ".join(ts)
            if 25 <= len(joined.split()) <= 70 and not set(us[:3]) & set(sents):
                paras.append(("+".join(us[:3]), joined))
    paras = evenly(paras, N_PARA)
    return [(u, "sent", good[u]) for u in sents] + [(u, "para", t) for u, t in paras]


def mandarin():
    tr = {}
    for line in (AISHELL / "transcript" / "aishell_transcript_v0.8.txt").read_text("utf-8").splitlines():
        uid, *words = line.split()
        tr[uid] = "".join(words)
    test_ids = sorted(p.stem for p in (AISHELL / "wav" / "test").rglob("*.wav"))
    test = [u for u in test_ids if u in tr]
    good = [u for u in test if 12 <= len(tr[u]) <= 25 and re.fullmatch(r"[一-鿿]+", tr[u])]
    sents = evenly(good, N_SENT)
    by_spk = collections.defaultdict(list)
    for u in test:
        by_spk[u[6:11]].append(u)  # BAC009S0764W0121 -> S0764
    paras = []
    for spk in sorted(by_spk):
        us = sorted(by_spk[spk])
        for i in range(0, len(us) - 2, 3):
            trip = us[i : i + 3]
            if set(trip) & set(sents):
                continue
            if all(re.fullmatch(r"[一-鿿]+", tr[u]) and 10 <= len(tr[u]) <= 25 for u in trip):
                paras.append(("+".join(trip), "。".join(tr[u] for u in trip) + "。"))
                break
    paras = evenly(paras, N_PARA)
    return [(u, "sent", tr[u] + "。") for u in sents] + [(u, "para", t) for u, t in paras], test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out)

    # reference clips: one English and one Mandarin utterance, plus a cloning prompt each
    # English: first utterance (sorted id) of 4-10 s from speaker 1089 (male) and 121 (female)
    texts, wavs = load_libritts(("1089_134686_", "121_121726_"))
    en_clips = []
    for prefix in ("1089_134686_", "121_121726_"):
        for uid in sorted(u for u in wavs if u.startswith(prefix)):
            b = wavs[uid]
            sr = struct.unpack("<I", b[24:28])[0]
            dur = (len(b) - 44) / (2 * sr)
            if 4.0 <= dur <= 10.0:
                en_clips.append(uid)
                break
    en_rows = english(texts)
    zh_rows, zh_test = mandarin()

    for name, rows in (("en.tsv", en_rows), ("zh.tsv", zh_rows)):
        with open(out / name, "w", encoding="utf-8") as f:
            f.write("id\tkind\ttext\n")
            for uid, kind, text in rows:
                f.write(f"{uid}\t{kind}\t{text}\n")

    (INPUTS / "libritts_r").mkdir(parents=True, exist_ok=True)
    (INPUTS / "aishell1").mkdir(parents=True, exist_ok=True)
    prompts = []
    for uid in en_clips:
        if uid not in wavs:
            raise SystemExit(f"{uid} not found in {LIBRITTS_TAR}")
        p = INPUTS / "libritts_r" / f"{uid}.wav"
        p.write_bytes(wavs[uid])
        prompts.append((uid, "en", str(p.relative_to(paths.BOOK_DATA)), texts[uid]))
    tr = {l.split()[0]: "".join(l.split()[1:]) for l in
          (AISHELL / "transcript" / "aishell_transcript_v0.8.txt").read_text("utf-8").splitlines()}
    for uid in ["BAC009S0764W0121", "BAC009S0768W0178"]:
        src = next((AISHELL / "wav" / "test").rglob(f"{uid}.wav"))
        p = INPUTS / "aishell1" / f"{uid}.wav"
        if not p.exists():
            p.symlink_to(src)  # a symlink to a single file in the store; nothing is created under it
        prompts.append((uid, "zh", str(p.relative_to(paths.BOOK_DATA)), tr[uid]))
    with open(out / "prompts.tsv", "w", encoding="utf-8") as f:
        f.write("id\tlang\tpath\ttext\n")
        for row in prompts:
            f.write("\t".join(row) + "\n")
    print(f"en: {len(en_rows)} rows, zh: {len(zh_rows)} rows, prompts: {len(prompts)}")


if __name__ == "__main__":
    main()
