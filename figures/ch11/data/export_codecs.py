"""Export data/codecs.json for bitrate_distance.py from code/tts/results/ch11_codecs.json
(code/tts/ch11_codecs/codec_roundtrip.py).

    python figures/ch11/data/export_codecs.py [--out PATH]
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
KEYS = ("codec", "setting", "codebooks", "frame_rate_hz", "kbit_per_s", "logmel_l1_db_mean", "encode_rtf_mean",
        "decode_rtf_mean")
CODEC_REPOS = ("kyutai/mimi", "facebook/encodec_24khz", "descript/dac_24khz")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "codecs.json"))
a = ap.parse_args()

r = json.load(open(REPO / "code" / "tts" / "results" / "ch11_codecs.json"))
out = {
    "source": "code/tts/results/ch11_codecs.json (code/tts/ch11_codecs/codec_roundtrip.py --threads 4, i5-12600K)",
    "models": [m for m in r["runinfo"]["models"] if m["repo"] in CODEC_REPOS],
    "rows": [{k: t[k] for k in KEYS} for t in r["table"]],
}
json.dump(out, open(a.out, "w"), indent=1)
print(len(out["rows"]), "rows ->", a.out)
