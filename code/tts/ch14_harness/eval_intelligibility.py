"""Chapter 14: intelligibility of synthesized speech through ASR, Seed-TTS-eval style.

English : openai/whisper-large-v3 (transformers), language forced to English, greedy.
          WER after Whisper's EnglishTextNormalizer on both sides.
Mandarin: funasr/paraformer-zh (FunASR). CER after hbtts.errorrate.normalize_zh on both sides.
Rates are pooled per group: total edits / total reference words (or characters).

Sources (repeat --source kind:path):
  kokoro:<rows.jsonl>       Kokoro run, exp=sentences (group = config) and paragraphs (group = config/mode)
  qwen:<rows.jsonl>         Qwen3-TTS client run (group = mode/rate, or concurrency level)
  human                     the original recordings of the 40 text-set sentences (LibriTTS-R, AISHELL-1),
                            the ASR floor for this text set

Run (GPU; code/tts/ch14_harness/run_eval_gpu.sh gives every --source of the book):
  python ch14_harness/eval_intelligibility.py --name ch14_intelligibility \
      --source human --source kokoro:$BOOK_DATA/runs/ch12_kokoro_cpu/rows.jsonl --source qwen:...
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hbtts import paths  # noqa: E402
from hbtts.errorrate import cer, pooled, wer  # noqa: E402
from hbtts.hub import rev  # noqa: E402
from hbtts.runinfo import hf_snapshot, run_dir, runinfo, write_json  # noqa: E402

TTS = Path(__file__).resolve().parents[1]
WHISPER = ("openai/whisper-large-v3", rev("openai/whisper-large-v3"))
PARAFORMER = ("funasr/paraformer-zh", rev("funasr/paraformer-zh"))
AISHELL_WAV = Path(f"{paths.BOOK_DATASETS}/aishell1/wav/test")
LIBRITTS_TAR = Path(f"{paths.BOOK_DATASETS}/libritts-r/test_clean.tar.gz")


def items_human() -> list[dict]:
    out = []
    en = [r for r in csv.DictReader(open(TTS / "textset" / "en.tsv", encoding="utf-8"), delimiter="\t")
          if r["kind"] == "sent"]
    want = {r["id"]: r for r in en}
    dst = run_dir("ch14_intelligibility") / "human_en"
    dst.mkdir(exist_ok=True)
    with tarfile.open(LIBRITTS_TAR, "r:gz") as tf:
        for m in tf:
            n = Path(m.name).name
            if n.endswith(".wav") and n[:-4] in want:
                p = dst / n
                p.write_bytes(tf.extractfile(m).read())
                out.append({"wav": str(p), "text": want[n[:-4]]["text"], "lang": "en", "group": "human/en",
                            "id": n[:-4]})
    for r in csv.DictReader(open(TTS / "textset" / "zh.tsv", encoding="utf-8"), delimiter="\t"):
        if r["kind"] == "sent":
            out.append({"wav": str(next(AISHELL_WAV.rglob(r["id"] + ".wav"))), "text": r["text"], "lang": "zh",
                        "group": "human/zh", "id": r["id"]})
    return out


def items_kokoro(path: str) -> list[dict]:
    out = []
    for line in open(path):
        r = json.loads(line)
        if not r.get("wav"):
            continue
        lang = "en" if r["config"] == "en" else "zh"
        g = f"kokoro/{r['config']}/sent" if r["exp"] == "sentences" else f"kokoro/{r['config']}/para/{r['mode']}"
        out.append({"wav": r["wav"], "text": r["text"], "lang": lang, "group": g, "id": r["id"]})
    return out


def items_qwen(path: str, prefix: str) -> list[dict]:
    out = []
    for line in open(path):
        r = json.loads(line)
        if not r.get("wav") or r.get("error"):
            continue
        if "concurrency" in r:
            g = f"{prefix}/c{r['concurrency']}"
        else:
            g = f"{prefix}/{r['mode']}/rate{r['rate'] or 0}"
        out.append({"wav": r["wav"], "text": r["text"], "lang": r["lang"], "group": g + "/" + r["lang"],
                    "id": r["id"]})
    return out


def load16k(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ch14_intelligibility")
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--cpu", action="store_true", help="run both ASR models on the CPU")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    items = []
    for s in args.source:
        kind, _, path = s.partition(":")
        if kind == "human":
            items += items_human()
        elif kind == "kokoro":
            items += items_kokoro(path)
        elif kind.startswith("qwen"):
            items += items_qwen(path, kind)
        else:
            raise SystemExit(f"unknown source {s}")
    dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    torch.set_num_threads(args.threads)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(WHISPER[0], revision=WHISPER[1])
    wmodel = WhisperForConditionalGeneration.from_pretrained(WHISPER[0], revision=WHISPER[1],
                                                             torch_dtype=dtype).to(dev).eval()
    norm = proc.tokenizer.normalize  # Whisper's EnglishTextNormalizer

    from funasr import AutoModel
    from huggingface_hub import snapshot_download
    pmodel = AutoModel(model=snapshot_download(PARAFORMER[0], revision=PARAFORMER[1]), device=dev,
                       disable_update=True, disable_pbar=True, log_level="ERROR")

    rows = []
    for it in items:
        x = load16k(it["wav"])
        if it["lang"] == "en":
            long = len(x) > 30 * 16000  # Whisper's window; longer clips use sequential long-form decoding
            enc = proc(x, sampling_rate=16000, return_tensors="pt",
                       **({"truncation": False, "padding": "longest", "return_attention_mask": True} if long else {}))
            feats = enc.input_features.to(dev, dtype)
            kw = {"attention_mask": enc.attention_mask.to(dev), "return_timestamps": True} if long else {}
            with torch.inference_mode():
                ids = wmodel.generate(feats, language="en", task="transcribe", num_beams=1, **kw)
            hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            e = wer(it["text"], hyp, norm)
        else:
            hyp = pmodel.generate(input=x)[0]["text"]
            e = cer(it["text"], hyp)
        rows.append({**it, "hyp": hyp, **e})
        print(json.dumps({"group": it["group"], "id": it["id"], "rate": round(e["rate"], 3), "hyp": hyp},
                         ensure_ascii=False), flush=True)

    out = run_dir(args.name)
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    table = [{"group": g, "lang": rs[0]["lang"], "metric": "WER" if rs[0]["lang"] == "en" else "CER",
              "n_utts": len(rs), "n_ref_units": sum(r["n_ref"] for r in rs), "pooled_rate": pooled(rs),
              "utts_with_errors": sum(1 for r in rs if r["sub"] + r["del"] + r["ins"] > 0)}
             for g, rs in sorted(groups.items())]
    summary = {"what": "ASR intelligibility (WER en / CER zh) of synthesized and original speech",
               "runinfo": runinfo(["torch", "torchaudio", "transformers", "funasr", "numpy"],
                                  {"asr_models": [hf_snapshot(WHISPER[0]), {"repo": PARAFORMER[0],
                                                                            "revision": PARAFORMER[1]}],
                                   "sources": args.source,
                                   "normalization": {"en": "Whisper EnglishTextNormalizer on ref and hyp",
                                                     "zh": "keep CJK, Latin, digits; drop punctuation/spaces"},
                                   "device": dev, "dtype": str(dtype),
                                   "decoding": {"whisper": "greedy, language=en",
                                                "paraformer": "FunASR default, no VAD, no punctuation model"}}),
               "table": table, "raw_dir": str(out)}
    write_json(TTS / "results" / f"{args.name}.json", summary)
    for t in table:
        print(json.dumps(t))


if __name__ == "__main__":
    main()
