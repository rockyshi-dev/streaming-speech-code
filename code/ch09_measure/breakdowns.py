"""Chapter 9: numbers the chapter derives from the raw per-utterance logs of the GPU-track runs.

    python code/ch09_measure/breakdowns.py      (ssh-asr env, after run_gpu_all.sh; CPU only)

1. Word latency of the FastConformer transducer on LibriSpeech test-clean, split in two for each
   att_context_size: the wait for the audio of the encoder step whose chunk contains the word's end
   (the step runs when the 160 ms packet holding its last feature frame has arrived), and the time after
   that step until the word appears for good. Also: the share of words that appear one or more steps
   (at least 160 ms) after that step, the words that became final only at the end-of-file flush, and the
   median compute per packet.
2. Nemotron 3.5 ASR on AISHELL-1: utterances whose final text contains Thai characters (U+0E00-U+0E7F).
Writes results/breakdowns.json.
"""
import json
import math
import os

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import align_ref, evaluate, latency, metrics, paths, provenance, textnorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = f"{paths.BOOK_DATA}/runs/"
MAN = "librispeech-test-clean-200"
# chunk geometry in feature frames (first step, later steps) for 8x subsampling, chunk = right + 1 frames
GEO = {"70_13": (105, 112), "70_6": (49, 56), "70_1": (9, 16), "70_0": (1, 8)}
tok = lambda s: textnorm.words(textnorm.EN_DEFAULT(s))


def step_time(frame, first, rest):
    """Arrival time (a 160 ms packet boundary) of the encoder step whose chunk contains feature `frame`."""
    last = first - 1 if frame < first else first + rest * ((frame - first) // rest + 1) - 1
    need = last * 160 + 256          # samples needed to complete that feature frame (centered STFT)
    return math.ceil(need / 2560) * 0.16


res = {"provenance": provenance.collect({"manifest": MAN}), "word_latency_split": {}, "nemotron_thai": {}}
fa = align_ref.load(MAN)
for ctx, (first, rest) in GEO.items():
    logs = evaluate.read_logs(RUNS + f"nemo-fastconformer-en-multi-{MAN}-rnnt-ctx{ctx}/utts.jsonl")
    lag, wait, after, flush_only, comp = [], [], [], 0, []
    for lg in logs:
        words = fa.get(lg.utt_id)
        ref_u = [textnorm.EN_DEFAULT(w) for w, _, _ in words]
        hyp_u = tok(lg.final_text)
        times = [None] * len(hyp_u)
        for i, j in metrics.matched_pairs(ref_u, hyp_u):
            times[j] = words[i][2]
        for t, l in zip(times, latency.emission_lags(lg, times, tok)):
            if t is None or l is None:
                continue
            st = step_time(int(round(t * 100)), first, rest)
            lag.append(l)
            wait.append(st - t)
            after.append(t + l - st)
            flush_only += abs(t + l - lg.final_done_s) < 1e-9
        comp += [e.compute_s for e in lg.events if e.compute_s > 0.005]
    lag, wait, after = map(np.array, (lag, wait, after))
    res["word_latency_split"][f"[{ctx.replace('_', ',')}]"] = {
        "words": int(lag.size), "word_latency_p50_ms": round(1000 * np.median(lag)),
        "wait_for_audio_p50_ms": round(1000 * np.median(wait)), "after_step_p50_ms": round(1000 * np.median(after)),
        "share_one_or_more_packets_late": round(float(np.mean(after >= 0.16 - 0.03)), 3),
        "final_only_at_flush": int(flush_only), "compute_per_packet_p50_ms": round(1000 * np.median(comp), 1)}
for ctx in ["56_13", "56_6", "56_3", "56_0"]:
    rows = [json.loads(l) for l in open(RUNS + f"nemo-nemotron-3.5-aishell1-test-200-rnnt-ctx{ctx}/utts.jsonl")]
    res["nemotron_thai"][f"[{ctx.replace('_', ',')}]"] = {
        "utterances": len(rows), "with_thai": sum(any("฀" <= c <= "๿" for c in r["final_text"]) for r in rows)}
provenance.write_json(os.path.join(HERE, "results", "breakdowns.json"), res)
print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))
