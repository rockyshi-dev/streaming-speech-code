"""Chapter 6: contextual biasing with sherpa-onnx hotwords (modified beam search, CPU).

    python code/ch06_decoding/hotwords.py --manifest librispeech-test-other-200
    python code/ch06_decoding/hotwords.py --manifest aishell1-test-200

Needs the modified_beam_search baseline from ch09_measure/run_cpu_all.sh (with the same --runs-suffix and
--tail-padding-s when a variant is run). The hotword list is the
set of reference words the baseline got wrong (substituted or deleted) that are at least 6 letters
long (English) or at least 2 characters long (Mandarin): a stand-in for a list a product already has,
such as contact names or product terms. This is an oracle list built from the test set, so the gain on
the target words is an upper bound; the number to watch is what the list does to the utterances that
do not contain any listed word.

Reports, with and without the list (same decoder, same beam):
  target-word hits   occurrences of listed words recognized correctly / all occurrences in the references
  WER/CER on utterances that contain a listed word, and on those that do not.
Writes results/hotwords-<manifest>.json, the list to runs/ch06/, raw logs to runs/<name>/.
"""
import argparse
import json
import os

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import evaluate, metrics, provenance, textnorm  # noqa: E402
from hbasr.backends_sherpa import TAIL_PADDING_S, SherpaConfig, SherpaRecognizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = evaluate.RUNS
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--score", type=float, default=1.5)
ap.add_argument("--tail-padding-s", type=float, default=TAIL_PADDING_S,
                help="synthetic silence appended before input_finished() (default: sherpa-onnx's own 0.66 s)")
ap.add_argument("--runs-suffix", default="",
                help="appended to the raw-output names under $BOOK_DATA/runs/; the baseline is read with the same suffix")
ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
args = ap.parse_args()
lang = evaluate.lang_of(args.manifest)
model = "zipformer-zh" if lang == "zh" else "zipformer-en"
base_name = f"sherpa-{model}-{args.manifest}-modified_beam_search"
base_logs = evaluate.read_logs(os.path.join(RUNS, base_name + args.runs_suffix, "utts.jsonl"))

norm = textnorm.EN_DEFAULT if lang == "en" else textnorm.ZH_DEFAULT


def ref_words(ref):  # Mandarin references are word-segmented with spaces: keep those words
    return norm(ref).split() if lang == "en" else [w for w in textnorm.ZH["nospace+punct"](ref).split()]



targets = set()
for lg in base_logs:
    if lang == "en":
        r, h = ref_words(lg.extra["ref"]), norm(lg.final_text).split()
        for i in metrics.aligned_ref_errors(r, h):
            if len(r[i]) >= 6:
                targets.add(r[i])
    else:
        rw = textnorm.ZH["raw"](lg.extra["ref"]).split()
        hyp = norm(lg.final_text)
        for w in rw:
            w = norm(w)
            if len(w) >= 2 and w not in hyp:
                targets.add(w)
targets = sorted(targets)
os.makedirs(os.path.join(RUNS, "ch06"), exist_ok=True)
hw_path = os.path.join(RUNS, "ch06", f"hotwords-{args.manifest}{args.runs_suffix}.txt")
with open(hw_path, "w", encoding="utf-8") as f:
    for w in targets:
        f.write((w.upper() if lang == "en" else w) + "\n")

name = base_name + "-hotwords" + args.runs_suffix
cfg = SherpaConfig(model=model, decoding_method="modified_beam_search", hotwords_file=hw_path,
                   hotwords_score=args.score, tail_padding_s=args.tail_padding_s)
rec = SherpaRecognizer(cfg)
path = evaluate.stream_corpus(rec, args.manifest, name)
hw_logs = evaluate.read_logs(path)


def analyse(logs):
    tok = textnorm.words if lang == "en" else textnorm.zh_units
    hits = total = 0
    with_t, without_t = [], []
    for lg in logs:
        ref, hyp = lg.extra["ref"], lg.final_text
        nref, nhyp = norm(ref), norm(hyp)
        present = [w for w in targets if (w in nref.split() if lang == "en" else w in nref)]
        for w in present:
            n_ref = nref.split().count(w) if lang == "en" else nref.count(w)
            n_hyp = nhyp.split().count(w) if lang == "en" else nhyp.count(w)
            total += n_ref
            hits += min(n_ref, n_hyp)
        (with_t if present else without_t).append((ref, hyp))
    return {"target_hits": hits, "target_occurrences": total,
            "with_targets": metrics.corpus_rate(with_t, tok, norm).to_dict() | {"utterances": len(with_t)},
            "without_targets": metrics.corpus_rate(without_t, tok, norm).to_dict() | {"utterances": len(without_t)}}


res = {"provenance": provenance.collect({"manifest": args.manifest, "recognizer": cfg.to_dict()}),
       "hotwords": targets, "n_hotwords": len(targets), "hotwords_score": args.score,
       "normalizer": norm.describe(),
       "baseline_modified_beam_search": analyse(base_logs), "with_hotwords": analyse(hw_logs),
       "raw": {"baseline": os.path.join(RUNS, base_name + args.runs_suffix), "hotwords": os.path.dirname(path)}}
res["latency_with_hotwords"] = evaluate.score(path, lang)["latency"]
provenance.write_json(os.path.join(args.results_dir, f"hotwords-{args.manifest}.json"), res)
print(json.dumps({k: res[k] for k in ("n_hotwords", "baseline_modified_beam_search", "with_hotwords")}, indent=1))
