"""Measured on a CPU: bitrate against reconstruction distance (mean absolute log-mel difference, dB,
80 bins) for Mimi (8/16/32 codebooks), EnCodec 24 kHz (1.5/6/24 kbit/s) and DAC 24 kHz (8/16/32
codebooks), averaged over two English and two Mandarin clips. Data: data/codecs.json, from
code/tts/results/ch11_codecs.json."""
import json
import pathlib

from figures.bookstyle import setup, figure, save

C, R = setup()
d = json.load(open(pathlib.Path(__file__).parent / "data" / "codecs.json"))
fig, ax = figure(aspect=0.45)
for k, codec in enumerate(["mimi", "encodec", "dac"]):
    pts = [r for r in d["rows"] if r["codec"] == codec]
    ax.plot([r["kbit_per_s"] for r in pts], [r["logmel_l1_db_mean"] for r in pts], "o-",
            color=C.series[k], label={"mimi": "Mimi (12.5 Hz)", "encodec": "EnCodec 24 kHz (75 Hz)",
                                     "dac": "DAC 24 kHz (75 Hz)"}[codec])
    for r in pts:
        ax.annotate(f"{r['codebooks']}", (r["kbit_per_s"], r["logmel_l1_db_mean"]), textcoords="offset points",
                    xytext=(4, 5), fontsize=8)
ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
ax.set_xlabel("bitrate (kbit/s, log scale); labels: codebooks per frame")
ax.set_ylabel("log-mel distance (dB)")
ax.legend(fontsize=9, frameon=False)
save(fig)
