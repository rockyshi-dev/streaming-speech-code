"""The endpointing trade-off, measured on 100 utterances per language with 2 s of near-silence appended:
p50 endpoint delay against the share of utterances cut off at an internal pause, for sherpa-onnx's rule 2
(trailing-silence setting 0.2 to 1.2 s) and Silero VAD (minimum silence 100 to 1,000 ms). English is
measured from the end of the last word in a forced alignment; Mandarin from sherpa's side only, against
the end of speech found by offline Silero VAD. Data: data/endpoint_sweep.json."""
import json
import pathlib

from figures.bookstyle import setup, figure, save

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "endpoint_sweep.json"))
fig, (a, b) = figure("wide", aspect=0.4, ncols=2, sharey=True)
lines = [(a, d["en_forced_alignment"]["sherpa_rule2_s"], "sherpa-onnx rule 2", C.series[0], "s"),
         (a, d["en_forced_alignment"]["silero_min_silence_ms"], "Silero VAD", C.series[1], "ms"),
         (b, d["zh_vad_reference"]["sherpa_rule2_s"], "sherpa-onnx rule 2", C.series[0], "s")]
for ax, pts, name, col, unit in lines:
    pts = [p for p in pts if p["p50_ms"] is not None]
    ax.plot([p["p50_ms"] for p in pts], [p["false_cutoffs"] for p in pts], "o-", color=col, label=name)
    for p in (pts[0], pts[-1]):
        ax.annotate(f"{p['setting']} {unit}", (p["p50_ms"], p["false_cutoffs"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
a.set_title("English (from the last word's end)", loc="left", fontsize=10)
b.set_title("Mandarin (from the VAD's end of speech)", loc="left", fontsize=10)
for ax in (a, b):
    ax.set_xlabel("endpoint delay, p50 (ms)")
    ax.set_xlim(0, 2300)
a.set_ylabel("utterances cut off early (of 100)")
a.legend(fontsize=9, frameon=False)
save(fig)
