"""Export data/qwen3tts_modes.json for qwen3tts_modes.py from code/tts/results/ch12_qwen3tts_input_modes.json
(code/tts/gpu/qwen3tts_client.py input-modes, run by code/tts/gpu/run_all_gpu.sh).

    python figures/ch12/data/export_qwen3tts_modes.py [--out PATH]
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
STATS = ("ttfa_s", "first_pcm_s", "rtf", "first_sentence_text_complete_s", "text_done_s")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "qwen3tts_modes.json"))
a = ap.parse_args()

r = json.load(open(REPO / "code" / "tts" / "results" / "ch12_qwen3tts_input_modes.json"))
out = {
    "source": "code/tts/results/ch12_qwen3tts_input_modes.json (vLLM-Omni 0.32.0, Qwen3-TTS-12Hz-1.7B-CustomVoice, "
              "one RTX 4090)",
    "rows": [{"mode": t["mode"], "text_rate_units_per_s": t["text_rate_units_per_s"], "lang": t["lang"], "n": t["n"]}
             | {s: {"n": t[s]["n"], "p50": t[s].get("p50"), "p95": t[s].get("p95")} for s in STATS}
             for t in r["table"]],
}
json.dump(out, open(a.out, "w"), indent=1)
print(len(out["rows"]), "rows ->", a.out)
