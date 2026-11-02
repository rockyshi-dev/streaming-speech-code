"""Chapter 9: collect every summary in results/ into one table (results/summary_table.json) and print
it as Markdown.

    python code/ch09_measure/make_tables.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
    if p.endswith(("summary_table.json", "-sweep.json")):
        continue
    d = json.load(open(p))
    if "accuracy" not in d:
        continue
    prov = d["provenance"]
    rec = prov.get("recognizer", {})
    lang = d["lang"]
    key = "lower+punct" if lang == "en" else "full"
    lat = d["latency"]
    rows.append({
        "run": os.path.basename(p)[:-5],
        "track": prov.get("track"),
        "hardware": prov.get("gpu_torch") or prov.get("cpu"),
        "model": rec.get("model"),
        "decoder": rec.get("decoding_method") or rec.get("decoder"),
        "att_context_size": rec.get("att_context_size"),
        "manifest": prov.get("manifest"),
        "metric": "WER" if lang == "en" else "CER",
        "error_rate_pct": round(100 * d["accuracy"][key]["rate"], 2),
        "error_rate_raw_pct": round(100 * d["accuracy"]["raw"]["rate"], 2),
        "normalizer": d["accuracy"][key]["normalizer"]["name"],
        "UPWR": round(d["stability"]["UPWR"], 3),
        "UPSR": round(d["stability"]["UPSR"], 3),
        "first_token_p50_ms": round(1000 * lat["first_token_delay_s"]["p50"]),
        "first_token_p90_ms": round(1000 * lat["first_token_delay_s"]["p90"]),
        "final_after_speech_end_p50_ms": round(1000 * lat["final_latency_after_speech_end_s"]["p50"]),
        "final_after_speech_end_p90_ms": round(1000 * lat["final_latency_after_speech_end_s"]["p90"]),
        "final_after_input_end_p50_ms": round(1000 * lat["final_latency_after_input_end_s"]["p50"]),
        "word_latency_fa_p50_ms": (round(1000 * d["word_emission_latency_vs_forced_alignment_s"]["p50"])
                                   if "word_emission_latency_vs_forced_alignment_s" in d else None),
        "word_latency_fa_p90_ms": (round(1000 * d["word_emission_latency_vs_forced_alignment_s"]["p90"])
                                   if "word_emission_latency_vs_forced_alignment_s" in d else None),
        "corpus_rtf": round(lat["corpus_rtf"], 4),
        "utterances": d["utterances"],
    })
json.dump(rows, open(os.path.join(HERE, "results", "summary_table.json"), "w"), indent=1, ensure_ascii=False)
cols = ["run", "metric", "error_rate_pct", "error_rate_raw_pct", "UPWR", "UPSR", "first_token_p50_ms",
        "first_token_p90_ms", "word_latency_fa_p50_ms", "word_latency_fa_p90_ms",
        "final_after_speech_end_p50_ms", "final_after_speech_end_p90_ms", "corpus_rtf"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for r in rows:
    print("| " + " | ".join(str(r[c]) for c in cols) + " |")
