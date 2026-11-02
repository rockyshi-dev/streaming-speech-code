"""Print the measured tables (Markdown) from code/serve/results/*.json, as used in the chapters.

    python tools/tables.py            # all
    python tools/tables.py ch15       # one chapter
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def g(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def f(x, nd=0):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def ch15():
    for p in sorted(R.glob("ch15_*.json")):
        d = json.loads(p.read_text())
        print(f"\n### {p.name}  (chunk {d['chunk_ms']:.0f} ms, {d['duration_s_per_N']:.0f} s per N; raw {d['raw_dir']})\n")
        print("| N | sessions | partial p50 | partial p95 | final p50 | final p95 | drift ms/s | server CPU cores | GPU mem | WER en | CER zh |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in d["rows"]:
            print(f"| {r['streams']} | {r['sessions']} | {f(g(r,'partial_latency_ms','p50'),1)} | {f(g(r,'partial_latency_ms','p95'),1)} | "
                  f"{f(g(r,'final_latency_ms','p50'),1)} | {f(g(r,'final_latency_ms','p95'),1)} | {f(r.get('slope_ms_per_s_median'),2)} | "
                  f"{f(r.get('server_cpu_cores'),2)} | {(r.get('gpu_after') or '-').split(',')[0]} | "
                  f"{f(g(r,'error_rate','en','rate'),3)} | {f(g(r,'error_rate','zh','rate'),3)} |")


def ch17():
    p = R / "ch17_opus_delay.json"
    if p.exists():
        d = json.loads(p.read_text())
        print(f"\n### ch17_opus_delay.json (libopus {d['meta'].get('libopus')}, {d['meta'].get('bitrate_target')} b/s target, voip)\n")
        print("| fs | frame ms | look-ahead ms | measured delay ms | frame + look-ahead ms | kb/s | enc us | dec us | SNR dB |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in d["frames"]:
            print(f"| {r['fs']} | {r['frame_ms']} | {r['lookahead_ms']} | {r['measured_delay_ms']} | {r['algorithmic_ms']} | {r['kbps']} | "
                  f"{r['enc_us_p50']} | {r['dec_us_p50']} | {r['snr_db']} |")
        print("\n| loss (synthetic) | lost/packets | SNR PLC vs clean | SNR FEC vs clean | kb/s with FEC | kb/s without |")
        print("|---|---|---|---|---|---|")
        for r in d["loss_synthetic"]:
            print(f"| {r['loss_rate']} | {r['lost']}/{r['packets']} | {r['snr_db_plc_vs_clean']} | {r['snr_db_fec_vs_clean']} | "
                  f"{r['kbps_with_fec']} | {r['kbps_without_fec']} |")
    p = R / "ch17_jitter_synthetic.json"
    if p.exists():
        d = json.loads(p.read_text())
        print("\n### ch17_jitter_synthetic.json (SYNTHETIC network)\n")
        print("| scenario | <=1% late: UDP | TCP rto 200 | TCP fast | <=0.1% late: UDP | TCP rto 200 | TCP fast |")
        print("|---|---|---|---|---|---|---|")
        for k, s in d["scenarios"].items():
            m = lambda q, t: s[q][t]["median"]  # noqa: E731
            print(f"| {k} | " + " | ".join(str(m(q, t)) for q in ("depth_for_1pct_ms", "depth_for_0.1pct_ms")
                                            for t in ("udp", "tcp_rto200", "tcp_fast")) + " |")
    for p in sorted(R.glob("ch17_ws_*.json")):
        d = json.loads(p.read_text())
        print(f"\n{p.name}: codec {d['codec']}, {d['payload_kbps']} kb/s, network p50/p95 {g(d,'network_ms','p50')}/"
              f"{g(d,'network_ms','p95')} ms, capture-start-to-decoded p50/p95 {g(d,'capture_start_to_decoded_ms','p50')}/"
              f"{g(d,'capture_start_to_decoded_ms','p95')} ms, decode p50 {g(d,'decode_us','p50')} us")


def ch18():
    for p in sorted(R.glob("ch18_*.json")):
        d = json.loads(p.read_text())
        a = d["meta"]["args"]
        mode = a["endpoint"] + (f" {a['silence_ms']:.0f} ms" if a["endpoint"] == "silence" else f" (trigger {a['trigger_ms']:.0f}, fallback {a['fallback_ms']:.0f} ms)")
        print(f"\n### {p.name}: endpoint {mode}; {d['turns']} turns, premature {d['premature_endpoints']}, reasons {d['endpoint_reasons']}")
        print(f"ASR: en WER {g(d,'asr_error_rate','en','rate')}, zh CER {g(d,'asr_error_rate','zh','rate')}\n")
        print("| stage | all p50 | all p95 | en p50 | en p95 | zh p50 | zh p95 |")
        print("|---|---|---|---|---|---|---|")
        keys = ["endpoint_delay_ms", "asr_final_after_endpoint_ms", "llm_request_after_final_ms", "llm_first_token_ms", "sentence_aggregation_ms",
                "tts_first_audio_ms", "playout_ms", "voice_to_voice_ms"]
        for k in keys:
            row = [g(d, "budget_all", k, "p50"), g(d, "budget_all", k, "p95"),
                   g(d, "budget_by_lang", "en", k, "p50"), g(d, "budget_by_lang", "en", k, "p95"),
                   g(d, "budget_by_lang", "zh", k, "p50"), g(d, "budget_by_lang", "zh", k, "p95")]
            print(f"| {k} | " + " | ".join(f(x) for x in row) + " |")
        print(f"\nturns used {g(d,'budget_all','turns_used')}, underruns {g(d,'budget_all','underruns_total')} "
              f"in {g(d,'budget_all','turns_with_underrun')} turns")
        if "barge_in" in d:
            print("\nbarge-in: " + json.dumps(d["barge_in"]))


if __name__ == "__main__":
    which = sys.argv[1:] or ["ch15", "ch17", "ch18"]
    for w in which:
        globals()[w]()
