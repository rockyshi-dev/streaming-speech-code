"""Chapter 12 CPU track: Kokoro-82M with sentence-level streaming on the CPU.

Kokoro is not autoregressive: it turns one whole text segment into one whole waveform.
"Streaming" here means sentence-level output streaming: cut the text into sentences,
synthesize them one by one, and hand each sentence's audio to the player as soon as it exists.
So the first audio cannot exist before the whole first sentence has been synthesized.

Configurations
  en     hexgrad/Kokoro-82M (v1.0), lang 'a' (American English), voice af_heart
  zh     hexgrad/Kokoro-82M-v1.1-zh, lang 'z', voice zf_001
  zh10   hexgrad/Kokoro-82M (v1.0), lang 'z', voice zf_xiaobei (the v1.0 Mandarin voice, for comparison)

Experiments (all on the CPU, torch threads fixed by --threads)
  sentences   each of the 20 single sentences per language as one request:
              TTFA = synthesis time of the whole sentence (the only chunk), vs its length
  paragraphs  each 3-sentence paragraph, three ways:
                whole      one call for the whole paragraph, one chunk
                sentence   all text available at t=0, one chunk per sentence
                stream     text arrives unit by unit at --rate units/s (simulated LLM tokens);
                           a sentence is synthesized once its last unit has arrived and the
                           previous sentence is done
              TTFA, RTF, chunk durations, inter-arrival times and underruns of a player that
              starts at the first chunk.

Run
  python ch12_kokoro_cpu/kokoro_stream.py --threads 4
Raw : $BOOK_DATA/runs/ch12_kokoro_cpu/   (rows.jsonl, wav/)
Sum.: code/tts/results/ch12_kokoro_cpu.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hbtts.hub import kokoro_pipeline, kokoro_voice  # noqa: E402
from hbtts.runinfo import hf_snapshot, run_dir, runinfo, write_json  # noqa: E402
from hbtts.stream import StreamRecord, summarize  # noqa: E402
from hbtts.textstream import split_sentences, text_units  # noqa: E402

TTS = Path(__file__).resolve().parents[1]
SR = 24000
CONFIGS = {
    "en": {"repo": "hexgrad/Kokoro-82M", "lang_code": "a", "voice": "af_heart", "textset": "en"},
    "zh": {"repo": "hexgrad/Kokoro-82M-v1.1-zh", "lang_code": "z", "voice": "zf_001", "textset": "zh"},
    "zh10": {"repo": "hexgrad/Kokoro-82M", "lang_code": "z", "voice": "zf_xiaobei", "textset": "zh"},
}


def load_items(lang: str, kind: str) -> list[dict]:
    rows = csv.DictReader(open(TTS / "textset" / f"{lang}.tsv", encoding="utf-8"), delimiter="\t")
    return [r for r in rows if r["kind"] == kind]


class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.p = kokoro_pipeline(cfg["lang_code"], cfg["repo"])
        self.voice = kokoro_voice(cfg["repo"], cfg["voice"])

    def synth(self, text: str) -> np.ndarray:
        """Whole waveform for `text` (Kokoro may split long text internally; the parts are joined)."""
        parts = [r.audio.numpy() for r in self.p(text, voice=self.voice, split_pattern=None) if r.audio is not None]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def sentence_ready_times(text: str, sents: list[str], units: list[str], rate: float | None) -> list[float]:
    """Release time of the unit that completes each sentence (units go out at i / rate)."""
    if not rate:
        return [0.0] * len(sents)
    ends, pos = [], 0
    for s in sents:
        pos = text.index(s, pos) + len(s)
        ends.append(pos)
    unit_end = np.cumsum([len(u) for u in units])
    return [int(np.searchsorted(unit_end, e, side="left")) / rate for e in ends]


def run_sentence_stream(eng: Engine, text: str, rate: float | None, split: bool) -> tuple[StreamRecord, list]:
    """Synthesize `text` as chunks, on the real clock. Returns the record and per-chunk timings."""
    sents = split_sentences(text) if split else [text]
    units = text_units(text)
    ready = sentence_ready_times(text, sents, units, rate)
    rec = StreamRecord(sr=SR, t0=time.perf_counter())
    rec.t_text_done = ready[-1]
    chunks = []
    for s, r in zip(sents, ready):
        wait = rec.t0 + r - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        t_start = time.perf_counter()
        y = eng.synth(s)
        t_end = time.perf_counter()
        rec.add(t_end, y)
        chunks.append({"text": s, "text_ready_s": r, "synth_start_s": t_start - rec.t0,
                       "synth_s": t_end - t_start, "audio_s": len(y) / SR,
                       "chunk_rtf": (t_end - t_start) / (len(y) / SR) if len(y) else None})
    return rec, chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rate", type=float, default=20.0, help="text units per second for the stream mode")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--configs", default="en,zh,zh10")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = run_dir("ch12_kokoro_cpu")
    (out / "wav").mkdir(exist_ok=True)
    rows = []
    t_load = {}
    for name in args.configs.split(","):
        cfg = CONFIGS[name]
        t = time.perf_counter()
        eng = Engine(cfg)
        t_load[name] = time.perf_counter() - t
        for it in load_items(cfg["textset"], "sent")[:2]:
            eng.synth(it["text"])  # warm-up
        # 1) single sentences
        for it in load_items(cfg["textset"], "sent"):
            rec, chunks = run_sentence_stream(eng, it["text"], None, split=False)
            s = rec.summary()
            wav = out / "wav" / f"{name}_sent_{it['id']}.wav"
            sf.write(wav, rec.audio(), SR, subtype="PCM_16")
            rows.append({"config": name, "exp": "sentences", "mode": "whole", "id": it["id"], "text": it["text"],
                         "n_chars": len(it["text"]), "n_units": len(text_units(it["text"])), "wav": str(wav),
                         **{k: s[k] for k in ("ttfa_s", "first_pcm_s", "rtf", "audio_s", "leading_inaudible_s")},
                         "chunks": chunks})
            print(json.dumps({k: rows[-1][k] for k in ("config", "id", "n_units", "ttfa_s", "rtf", "audio_s")}),
                  flush=True)
        # 2) paragraphs
        for rep in range(args.repeats):
            for it in load_items(cfg["textset"], "para"):
                for mode, rate, split in (("whole", None, False), ("sentence", None, True),
                                          ("stream", args.rate, True)):
                    rec, chunks = run_sentence_stream(eng, it["text"], rate, split)
                    s = rec.summary()
                    row = {"config": name, "exp": "paragraphs", "mode": mode, "rate": rate, "repeat": rep,
                           "id": it["id"], "text": it["text"], **s, "arrivals": rec.arrivals,
                           "piece_samples": [len(p) for p in rec.pieces], "chunks": chunks}
                    if rep == 0:
                        wav = out / "wav" / f"{name}_para_{it['id'].split('+')[0]}_{mode}.wav"
                        sf.write(wav, rec.audio(), SR, subtype="PCM_16")
                        row["wav"] = str(wav)
                    rows.append(row)
                    print(json.dumps({k: row.get(k) for k in ("config", "mode", "repeat", "ttfa_s", "rtf",
                                                               "audio_s", "underruns", "n_pieces")}), flush=True)
        del eng

    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")

    table_sent, table_para = [], []
    for name in args.configs.split(","):
        rs = [r for r in rows if r["config"] == name and r["exp"] == "sentences"]
        x = np.array([r["n_units"] for r in rs], dtype=float)
        y = np.array([r["ttfa_s"] for r in rs], dtype=float)
        slope, icpt = np.polyfit(x, y, 1)
        table_sent.append({"config": name, "n": len(rs), "ttfa_s": summarize(y), "rtf": summarize([r["rtf"] for r in rs]),
                           "audio_s": summarize([r["audio_s"] for r in rs]),
                           "ttfa_vs_units_fit": {"slope_s_per_unit": slope, "intercept_s": icpt,
                                                 "pearson_r": float(np.corrcoef(x, y)[0, 1])}})
        for mode in ("whole", "sentence", "stream"):
            rp = [r for r in rows if r["config"] == name and r["exp"] == "paragraphs" and r["mode"] == mode]
            chunk_rtf = [c["chunk_rtf"] for r in rp for c in r["chunks"] if c["chunk_rtf"]]
            table_para.append({
                "config": name, "mode": mode, "rate": rp[0]["rate"], "n": len(rp),
                "ttfa_s": summarize([r["ttfa_s"] for r in rp]), "rtf": summarize([r["rtf"] for r in rp]),
                "first_chunk_audio_s": summarize([r["chunks"][0]["audio_s"] for r in rp]),
                "first_text_ready_s": summarize([r["chunks"][0]["text_ready_s"] for r in rp]),
                "chunk_audio_s": summarize([c["audio_s"] for r in rp for c in r["chunks"]]),
                "chunk_rtf": summarize(chunk_rtf),
                "underruns_total": int(sum(r["underruns"] for r in rp)),
                "stall_s": summarize([r["stall_s"] for r in rp]),
                "audio_s": summarize([r["audio_s"] for r in rp])})

    summary = {
        "what": "Kokoro-82M on the CPU: per-sentence TTFA, paragraph TTFA/RTF/underruns, three ways of streaming",
        "runinfo": runinfo(["torch", "kokoro", "misaki", "numpy", "spacy", "jieba", "pypinyin"],
                           {"torch_threads": args.threads, "text_rate_units_per_s": args.rate,
                            "repeats": args.repeats, "configs": CONFIGS, "load_s": t_load,
                            "models": [hf_snapshot("hexgrad/Kokoro-82M"), hf_snapshot("hexgrad/Kokoro-82M-v1.1-zh")],
                            "clock": "client and synthesis share one process; arrival = moment synthesis returns",
                            "player": "starts at the first chunk, no pre-buffer"}),
        "sentences": table_sent, "paragraphs": table_para, "raw_dir": str(out),
    }
    write_json(TTS / "results" / "ch12_kokoro_cpu.json", summary)
    print(json.dumps({"sentences": table_sent}, indent=1, default=float))


if __name__ == "__main__":
    main()
