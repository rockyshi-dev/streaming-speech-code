"""Corpus evaluation shared by the CPU (sherpa-onnx) and GPU (NeMo) runners (chapters 6 and 9).

`stream_corpus` streams every utterance of a manifest through a recognizer with the latency harness
and writes one JSON line per utterance (the full chunk log, final text, reference, VAD speech
region) to runs/<name>/utts.jsonl. `score` reads that file back and computes everything the
chapter reports: WER/CER under each stated normalizer, partial stability (UPWR/UPSR), and latency
percentiles.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

from . import align_ref, audio, data, latency, metrics, paths, textnorm, vad

RUNS = str(paths.book_data("runs"))


def lang_of(manifest: str) -> str:
    return "zh" if "aishell" in manifest else "en"


def vad_regions(manifest: str, utts: list[data.Utt]) -> dict:
    """Silero speech onset/offset per utterance, computed once per manifest and cached."""
    path = os.path.join(RUNS, "vad", manifest + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    out = {}
    for u in utts:
        x, sr = audio.read_wav(u.path)
        out[u.utt_id] = vad.onset_offset(x, sr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=0)
    return out


def stream_corpus(rec, manifest: str, name: str, chunk_ms: float = 160.0, limit: Optional[int] = None,
                  unit_times: Optional[Callable] = None, stop_at_endpoint: bool = False,
                  on_utt: Optional[Callable] = None) -> str:
    utts = data.read_manifest(manifest)[:limit]
    regions = vad_regions(manifest, data.read_manifest(manifest))
    out_dir = os.path.join(RUNS, name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "utts.jsonl")
    t0 = time.time()
    with open(path, "w", encoding="utf-8") as f:
        for i, u in enumerate(utts):
            x, sr = audio.read_wav(u.path)
            assert sr == rec.sample_rate
            sp = regions.get(u.utt_id)
            lg = latency.run_stream(rec, x, u.utt_id, chunk_ms, "simulated", speech=tuple(sp) if sp else None,
                                    stop_at_endpoint=stop_at_endpoint)
            lg.extra["ref"] = u.text
            if unit_times is not None:
                lg.extra["unit_times"] = unit_times(rec)
            if on_utt is not None:
                on_utt(rec, lg)
            f.write(lg.to_json() + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {name}: {i + 1}/{len(utts)} utterances, {time.time() - t0:.0f} s", flush=True)
    return path


def read_logs(path: str) -> list[latency.StreamLog]:
    return [latency.StreamLog.from_dict(json.loads(l)) for l in open(path, encoding="utf-8")]


def score(path: str, lang: str, manifest: Optional[str] = None) -> dict:
    logs = read_logs(path)
    norms = textnorm.EN if lang == "en" else textnorm.ZH
    tok = textnorm.words if lang == "en" else textnorm.zh_units
    default = textnorm.EN_DEFAULT if lang == "en" else textnorm.ZH_DEFAULT
    pairs = [(lg.extra["ref"], lg.final_text) for lg in logs]

    acc = {}
    for key, n in norms.items():
        acc[key] = metrics.corpus_rate(pairs, tok, n).to_dict() | {"normalizer": n.describe()}
    if lang == "zh":  # what a naive scorer that keeps the reference's word-separating spaces reports
        acc["raw_chars_incl_spaces"] = metrics.corpus_rate(pairs, textnorm.zh_units_with_spaces,
                                                           textnorm.ZH["raw"]).to_dict()

    stab = []
    for lg in logs:
        hyps = []
        for _, s in lg.emissions():
            u = tok(default(s))
            if not hyps or u != hyps[-1]:
                hyps.append(u)
        stab.append(metrics.stability(hyps or [[]]))

    summaries = [lg.summary() for lg in logs]
    lat = latency.aggregate(summaries)
    lags = []
    for lg in logs:
        if "unit_times" in lg.extra and lg.extra["unit_times"] is not None:
            lags += [v for v in latency.emission_lags(lg, lg.extra["unit_times"], lambda s: tok(default(s)))
                     if v is not None]
    fa = align_ref.load(manifest) if manifest else None
    word_lat, last_word = [], []
    if fa:
        for lg in logs:
            words = fa.get(lg.utt_id)
            if not words:
                continue
            ref_u = [default(w) for w, _, _ in words]
            hyp_u = tok(default(lg.final_text))
            times = [None] * len(hyp_u)
            for i, j in metrics.matched_pairs(ref_u, hyp_u):
                times[j] = words[i][2]  # reference word END time
            word_lat += [v for v in latency.emission_lags(lg, times, lambda s: tok(default(s))) if v is not None]
            last_word.append(lg.final_done_s - words[-1][2])
    out = {
        "utterances": len(logs),
        "audio_total_s": lat["audio_total_s"],
        "lang": lang,
        "scoring_units": "words" if lang == "en" else "CJK characters (embedded Latin words count as one unit)",
        "default_normalizer": default.name,
        "accuracy": acc,
        "stability": metrics.upwr_upsr(stab) | {"normalizer": default.name},
        "latency": lat,
        "raw": path,
    }
    if lags:
        out["emission_lag_vs_model_timestamps_s"] = metrics.percentiles(lags)
    if word_lat:
        out["word_emission_latency_vs_forced_alignment_s"] = metrics.percentiles(word_lat) | {
            "definition": "time a correctly recognized word first appears in a partial result that never "
                          "changes it again, minus that word's end time in a wav2vec 2.0 forced alignment "
                          "of the reference (20 ms frames)"}
        out["final_latency_after_last_aligned_word_s"] = metrics.percentiles(last_word)
    return out
