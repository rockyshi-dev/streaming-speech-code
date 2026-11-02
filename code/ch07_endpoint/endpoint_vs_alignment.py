"""Chapter 7: the endpoint sweep read against an independent end-of-speech reference (English).

    python code/ch07_endpoint/endpoint_vs_alignment.py      (after endpoint_sweep.py and hbasr.align_ref)

endpoint_sweep.py measures from the end of speech found by offline Silero VAD. For Silero's own
streaming endpointer that reference is the same model, so its delay comes out as the rule's silence
setting plus a constant. This script re-reads the same raw rows against the end of the last word in a
wav2vec 2.0 forced alignment of the reference transcript, which neither endpointer has seen, and
adds an `en_vs_forced_alignment` block to results/endpoint_sweep.json.
"""
import json
import os
from collections import defaultdict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import align_ref, metrics, paths, provenance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
summ_path = os.path.join(HERE, "results", "endpoint_sweep.json")
summ = json.load(open(summ_path))
fa = align_ref.load("librispeech-test-clean-200")
TOL = 0.05
groups = defaultdict(list)
for line in open(paths.resolve(summ["raw"]), encoding="utf-8"):
    r = json.loads(line)
    if r["lang"] == "en":
        groups[(r["system"], r["setting"])].append(r)
out = {}
for (system, setting), rows in sorted(groups.items()):
    delays, cut, none = [], 0, 0
    for r in rows:
        end = fa[r["utt_id"]][-1][2]
        if r["endpoint_s"] is None:
            none += 1
        elif r["endpoint_s"] < end - TOL:
            cut += 1
        else:
            delays.append(r["endpoint_s"] - end)
    out.setdefault(system, {})[str(setting)] = {"endpoint_delay_s": metrics.percentiles(delays),
                                                "false_cutoffs": cut, "no_endpoint": none, "utterances": len(rows)}
    print(system, setting, round(out[system][str(setting)]["endpoint_delay_s"]["p50"] or -1, 3), "cut", cut)
summ["en_vs_forced_alignment"] = out | {
    "reference": "end time of the last reference word, wav2vec 2.0 Base 960 h CTC forced alignment (20 ms)",
    "computed": provenance.collect()}
provenance.write_json(summ_path, summ)
