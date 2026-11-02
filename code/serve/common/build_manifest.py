"""Pick a fixed, reproducible set of English and Mandarin test utterances.

English: LibriSpeech test-clean (CC BY 4.0, https://www.openslr.org/12).
Mandarin: AISHELL-1 test (Apache-2.0, https://www.openslr.org/33).
Both are read in place from $BOOK_DATASETS. Selection: utterances of 3-12 s, ordered by the sha1 of their id,
first K per language. Paths are stored relative to $BOOK_DATASETS. The output is committed
(code/serve/data/test_utts.jsonl) so every run uses the same list.

    python -m common.build_manifest --per-lang 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf

from common import paths

LS = Path(f"{paths.BOOK_DATASETS}/librispeech/extracted/LibriSpeech/test-clean")
AS = Path(f"{paths.BOOK_DATASETS}/aishell1")
OUT = Path(__file__).resolve().parents[1] / "data" / "test_utts.jsonl"


def librispeech():
    for trans in sorted(LS.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            yield uid, trans.parent / f"{uid}.flac", text.strip()


def aishell():
    texts = {}
    for line in (AS / "transcript" / "aishell_transcript_v0.8.txt").read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            texts[parts[0]] = parts[1].replace(" ", "")
    for wav in sorted((AS / "wav" / "test").glob("*/*.wav")):
        if wav.stem in texts:
            yield wav.stem, wav, texts[wav.stem]


def pick(source, lang, k, lo=3.0, hi=12.0):
    cands = sorted(source, key=lambda r: hashlib.sha1(r[0].encode()).hexdigest())
    out = []
    for uid, path, text in cands:
        dur = sf.info(str(path)).duration
        if lo <= dur <= hi:
            rel = str(Path(path).relative_to(paths.BOOK_DATASETS))
            out.append({"id": uid, "lang": lang, "path": rel, "text": text, "duration": round(dur, 3)})
        if len(out) == k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=40)
    a = ap.parse_args()
    rows = pick(librispeech(), "en", a.per_lang) + pick(aishell(), "zh", a.per_lang)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {OUT}; total {sum(r['duration'] for r in rows):.1f} s")


if __name__ == "__main__":
    main()
