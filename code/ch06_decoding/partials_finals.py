"""Chapter 6: partial and final results, greedy vs modified beam search, on one English and one
Mandarin utterance (sherpa-onnx streaming Zipformer, CPU).

    python code/ch06_decoding/partials_finals.py

For each utterance and decoder, prints the "partial waterfall": every distinct partial result with
the time it became available (simulated real-time pacing, 160 ms chunks), the final result, and the
utterance's UPWR/UPSR. Greedy transducer search only ever appends tokens, so its word-level
revisions come from word pieces that grow into a longer word; beam search can also replace earlier
tokens when a different hypothesis overtakes the best one.
Writes results/partials_finals.json.
"""
import os

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, latency, metrics, paths, provenance, textnorm, vad  # noqa: E402
from hbasr.backends_sherpa import SherpaConfig, SherpaRecognizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [
    ("en", "zipformer-en", f"{paths.BOOK_DATASETS}/librispeech/extracted/LibriSpeech/test-other/1688/142285/1688-142285-0007.flac",
     "I REALLY WAS VERY MUCH AFRAID OF SHOWING HIM HOW MUCH SHOCKED I WAS AT SOME PARTS OF WHAT HE SAID"),
    ("zh", "zipformer-zh", f"{paths.BOOK_DATASETS}/aishell1/wav/test/S0764/BAC009S0764W0140.wav",
     "其中  广州  深圳  甚至  出现  了  多  个  日光  盘"),
]
res = {"provenance": provenance.collect(), "cases": []}
for lang, model, wav, ref in CASES:
    x, sr = audio.read_wav(wav)
    sp = vad.onset_offset(x, sr)
    norm = textnorm.EN_DEFAULT if lang == "en" else textnorm.ZH_DEFAULT
    tok = textnorm.words if lang == "en" else textnorm.zh_units
    for dec in ("greedy_search", "modified_beam_search"):
        rec = SherpaRecognizer(SherpaConfig(model=model, decoding_method=dec))
        lg = latency.run_stream(rec, x, os.path.basename(wav), 160, "simulated", speech=sp)
        ems = lg.emissions()
        hyps = []
        for _, s in ems:
            u = tok(norm(s))
            if not hyps or u != hyps[-1]:
                hyps.append(u)
        st = metrics.stability(hyps)
        case = {"lang": lang, "decoder": dec, "wav": wav, "ref": ref, "final": lg.final_text,
                "waterfall": [{"t_s": round(t, 3), "text": s} for t, s in ems],
                "unstable_units": st.unstable_units, "revisions": st.revisions, "final_units": st.final_units,
                "UPWR": st.unstable_units / max(st.final_units, 1), "summary": lg.summary()}
        res["cases"].append(case)
        print(f"--- {lang} {dec}: UPWR {case['UPWR']:.2f}, revisions {st.revisions}")
        for e in case["waterfall"]:
            print(f"  {e['t_s']:6.2f} s  {e['text']}")
provenance.write_json(os.path.join(HERE, "results", "partials_finals.json"), res)
