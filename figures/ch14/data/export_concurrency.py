"""Export data/concurrency.json for ttfa_vs_concurrency.py from code/tts/results/ch14_qwen3tts_concurrency.json
(code/tts/gpu/qwen3tts_client.py concurrency; both languages pooled).

    python figures/ch14/data/export_concurrency.py [--out PATH]
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "concurrency.json"))
a = ap.parse_args()

r = json.load(open(REPO / "code" / "tts" / "results" / "ch14_qwen3tts_concurrency.json"))
out = {
    "source": "code/tts/results/ch14_qwen3tts_concurrency.json (closed loop, sentences, English and Mandarin, "
              "one RTX 4090)",
    "rows": [{"concurrency": t["concurrency"], "n": t["n"], "ttfa_p50_s": t["ttfa_s"]["p50"],
              "ttfa_p95_s": t["ttfa_s"]["p95"], "first_pcm_p50_s": t["first_pcm_s"]["p50"], "rtf_p50": t["rtf"]["p50"],
              "rtf_p95": t["rtf"]["p95"], "requests_with_underrun": t["requests_with_underrun"],
              "audio_s_per_wall_s": t["audio_s_per_wall_s"]}
             for t in r["table"] if t["lang"] == "all"],
}
json.dump(out, open(a.out, "w"), indent=1)
print(len(out["rows"]), "rows ->", a.out)
