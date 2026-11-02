"""Read the cap-192 concurrency sweep against its pre-registered gates and decision rule.

    python code/tts/ch14_harness/check_cap192_gates.py --baseline <the rerun's cap-64 result file> \\
        --log <stage log with the two ECC lines> --server-log <server.log of the cap-192 run> \\
        [--result code/tts/results/ch14_qwen3tts_concurrency_cap192.json] [--package-yaml <installed qwen3_tts.yaml>]

Gates G0 to G7 are those written before the run (G5 as a before/after delta). A gate whose evidence
is missing counts as not passed, never as passed. The output is a table and `<result>.gates.json`. Standard library only.

  G0  the baseline is from the same GPU model as the result
  G1  the server log lists MTP graph sizes above 64 and the stored deploy text holds two `max_num_seqs: 192` lines
  G2  the stored deploy text differs from the package file in exactly the two `max_num_seqs` lines
  G3  control level 64 within 15% of the baseline's level 64; the repeat of 64 within 10% of the control
  G4  no failed request in the level, and no failed request in its warm-up burst
  G5  `DRAM Uncorrectable` is 0 before and after
  G6a no preemption / KV-full line in the server log inside the level's window (sees only the "fail" mode, see A1)
  G6b mean audio per request within 10% of the baseline's level-64 mean
  G6c the server's preemption counter did not move between the start of the level's warm-up and the end of the level
  G6d the share of requests whose audio length is further from that text's baseline median than the baseline's own
      99th-percentile deviation is at most 5% (the baseline's share is 1%)
  G7  client event-loop overshoot p99 below 50 ms
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
import time

TTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)")
TROUBLE = re.compile(r"preempt|evict|KV cache.*(full|exhaust)|out of memory|OutOfMemory", re.I)
ANOMALY_SHARE_MAX = 0.05


def within(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol * abs(b)


def level_label(r):
    return f"{r['concurrency']}" + (f" (visit {r['visit']})" if r.get("visit", 0) else "")


def g0(result, base):
    a = result["runinfo"].get("gpu", "")
    b = base["runinfo"].get("gpu", "")
    ka, kb = a.split(",")[0].strip(), b.split(",")[0].strip()
    return bool(ka) and ka == kb, f"result GPU '{ka}', baseline GPU '{kb}'"


def g1(result, server_log_text):
    sizes = [max(int(x) for x in m.split(",") if x.strip().isdigit())
             for m in re.findall(r"mtp graphs for sizes \[([0-9, ]+)\]", server_log_text)]
    deploy = result["runinfo"]["server"].get("deploy_config_text", "")
    n_192 = len(re.findall(r"^    max_num_seqs: 192$", deploy, re.M))
    ok = bool(sizes) and max(sizes) > 64 and n_192 == 2
    return ok, f"largest MTP graph size {max(sizes) if sizes else None}; max_num_seqs: 192 lines in stored deploy text: {n_192}"


def g2(result, package_yaml):
    if not package_yaml or not os.path.exists(package_yaml):
        return None, "package yaml not found; pass --package-yaml"
    a = open(package_yaml).read().splitlines()
    b = result["runinfo"]["server"].get("deploy_config_text", "").splitlines()
    if len(a) != len(b):
        return False, f"line counts differ: {len(a)} vs {len(b)}"
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    ok = len(diff) == 2 and all("max_num_seqs" in x and "max_num_seqs" in y for x, y in diff)
    return ok, f"{len(diff)} differing lines"


def g3(levels, base64):
    visits = [r for r in levels if r["concurrency"] == 64]
    if len(visits) < 2 or not all(v["n_ok"] for v in visits[:2]):
        return None, "needs two visits of level 64 with completed requests"
    first, second = visits[0], visits[-1]
    ctrl = (within(first["rtf"]["p50"], base64["rtf"]["p50"], 0.15)
            and within(first["ttfa_s"]["p50"], base64["ttfa_s"]["p50"], 0.15))
    rep = (within(second["rtf"]["p50"], first["rtf"]["p50"], 0.10)
           and within(second["ttfa_s"]["p50"], first["ttfa_s"]["p50"], 0.10))
    msg = (f"control vs baseline: RTF {first['rtf']['p50']:.3f} vs {base64['rtf']['p50']:.3f}, "
           f"TTFA {first['ttfa_s']['p50']:.3f} vs {base64['ttfa_s']['p50']:.3f}; "
           f"repeat vs control: RTF {second['rtf']['p50']:.3f}, TTFA {second['ttfa_s']['p50']:.3f}; repeat within 10%: {rep}")
    return ctrl, msg


def log_hits(server_log_text, t0, t1, year):
    n = 0
    stamped = 0
    for line in server_log_text.splitlines():
        line = ANSI.sub("", line)
        m = STAMP.search(line)
        if not m:
            continue
        mo, d, h, mi, s = (int(x) for x in m.groups())
        t = time.mktime((year, mo, d, h, mi, s, 0, 0, -1))
        if not (t0 - 1 <= t <= t1 + 1):
            continue
        stamped += 1
        if TROUBLE.search(line):
            n += 1
    return n if stamped else None  # no timestamped line inside the window: the window is not covered by the log


def gate_ecc(log_text):
    """No new uncorrectable DRAM error during the run: the reading after equals the reading before. (The counter is
    cumulative since the driver loaded, so after a card has logged one error an absolute zero can no longer be required.)"""
    vals = re.findall(r"DRAM Uncorrectable\s*:\s*(\d+)", log_text)
    return len(vals) >= 2 and vals[0] == vals[-1], f"readings before, after: {vals}"


def read_rows(path):
    return [json.loads(line) for line in open(path)] if path and os.path.exists(path) else []


def baseline_item_stats(base_rows):
    """Per text: median audio length of the baseline's successful requests; and the baseline's 99th-percentile
    relative deviation from those medians (the threshold of G6d)."""
    by = {}
    for r in base_rows:
        if not r.get("error"):
            by.setdefault((r["lang"], r["id"]), []).append(r["audio_s"])
    med = {k: statistics.median(v) for k, v in by.items()}
    dev = sorted(abs(r["audio_s"] - med[(r["lang"], r["id"])]) / med[(r["lang"], r["id"])]
                 for r in base_rows if not r.get("error") and (r["lang"], r["id"]) in med)
    thr = dev[min(len(dev) - 1, int(0.99 * len(dev)))] if dev else None
    return med, thr


def anomaly_share(rows, visit, med, thr):
    rs = [r for r in rows if r.get("visit") == visit and not r.get("error") and (r["lang"], r["id"]) in med]
    if not rs or thr is None:
        return None
    bad = sum(1 for r in rs if abs(r["audio_s"] - med[(r["lang"], r["id"])]) / med[(r["lang"], r["id"])] > thr)
    return bad / len(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=os.path.join(TTS, "results", "ch14_qwen3tts_concurrency_cap192.json"))
    ap.add_argument("--baseline", required=True, help="the rerun's cap-64 sweep (same card)")
    ap.add_argument("--server-log")
    ap.add_argument("--log", required=True, help="stage log holding the before/after 'DRAM Uncorrectable' lines")
    ap.add_argument("--package-yaml")
    ap.add_argument("--rows")
    ap.add_argument("--baseline-rows")
    args = ap.parse_args()
    if not args.package_yaml:
        found = glob.glob(os.path.expanduser("~/miniconda3/envs/ssh-tts-gpu/lib/python*/site-packages/vllm_omni/deploy/qwen3_tts.yaml"))
        args.package_yaml = found[0] if found else None
    result = json.load(open(args.result))
    base = json.load(open(args.baseline))
    rows = read_rows(args.rows or os.path.join(result.get("raw_dir", ""), "rows.jsonl"))
    base_rows = read_rows(args.baseline_rows or os.path.join(base.get("raw_dir", ""), "rows.jsonl"))
    server_log = open(args.server_log).read() if args.server_log and os.path.exists(args.server_log) else ""
    year = time.localtime(os.path.getmtime(args.server_log)).tm_year if server_log else time.localtime().tm_year

    base64 = next(r for r in base["table"] if r["lang"] == "all" and r["concurrency"] == 64)
    mine = [r for r in result["table"] if r["lang"] == "all"]
    med, thr = baseline_item_stats(base_rows)
    out = {"gates": {}, "levels": [], "anomaly_threshold_rel_dev": thr}

    out["gates"]["G0"] = g0(result, base)
    out["gates"]["G1"] = g1(result, server_log)
    out["gates"]["G2"] = g2(result, args.package_yaml)
    out["gates"]["G3"] = g3(mine, base64)
    out["gates"]["G5"] = gate_ecc(open(args.log).read())

    for r in mine:
        done = r["n_ok"] > 0
        p0, p1 = r.get("preemptions_start"), r.get("preemptions_end")
        g6a_n = log_hits(server_log, r["wall_start_epoch"], r["wall_end_epoch"], year) if server_log and "wall_start_epoch" in r else None
        share = anomaly_share(rows, r.get("visit"), med, thr)
        g7_p99 = (r.get("loop_overshoot_s") or {}).get("p99")
        out["levels"].append({
            "level": level_label(r), "N": r["concurrency"], "visit": r.get("visit", 0), "n": r["n"],
            "rtf_p50": r["rtf"].get("p50") if done else None, "ttfa_p50": r["ttfa_s"].get("p50") if done else None,
            "ttfa_p95": r["ttfa_s"].get("p95") if done else None,
            "requests_with_underrun": r["requests_with_underrun"], "audio_s_per_wall_s": r["audio_s_per_wall_s"],
            "mean_in_flight_est": r.get("mean_in_flight_est"),
            "G4_ok": done and r["n_ok"] == r["n"] and not r["errors"] and r.get("warmup_errors") == 0,
            "G6a_hits": g6a_n, "G6a_ok": None if g6a_n is None else g6a_n == 0,
            "G6b_mean_audio_s": r["audio_s"].get("mean") if done else None,
            "G6b_ok": within(r["audio_s"].get("mean") if done else None, base64["audio_s"]["mean"], 0.10),
            "G6c_preemptions": None if p0 is None or p1 is None else p1 - p0,
            "G6c_ok": None if p0 is None or p1 is None else p1 - p0 == 0,
            "G6d_anomaly_share": share, "G6d_ok": None if share is None else share <= ANOMALY_SHARE_MAX,
            "G7_loop_p99_ms": None if g7_p99 is None else 1000 * g7_p99, "G7_ok": None if g7_p99 is None else g7_p99 < 0.05,
        })

    def clean(lv):  # every per-level gate must be True; missing evidence (None) is not clean
        return all(lv[k] is True for k in ("G4_ok", "G6a_ok", "G6b_ok", "G6c_ok", "G6d_ok", "G7_ok"))

    seen, firsts = set(), []  # the first visit of each N; the repeat of 64 is a drift check, not a level of the sweep
    for lv in out["levels"]:
        if lv["N"] not in seen:
            seen.add(lv["N"])
            firsts.append(lv)
    firsts.sort(key=lambda x: x["N"])
    out["first_rtf_ge_1"] = next((lv["N"] for lv in firsts if lv["rtf_p50"] is not None and lv["rtf_p50"] >= 1), None)
    out["first_underrun"] = next((lv["N"] for lv in firsts if lv["requests_with_underrun"] > 0), None)
    out["first_level_failing_a_G6_or_G7_gate"] = next((lv["N"] for lv in firsts if not clean(lv)), None)
    by_n = {lv["N"]: lv for lv in firsts}
    need = [96, 128, 160, 192]
    if all(n in by_n and clean(by_n[n]) and by_n[n]["rtf_p50"] is not None for n in need):
        out["prediction_holds"] = by_n[96]["rtf_p50"] < 1 and all(by_n[n]["rtf_p50"] >= 1 for n in (128, 160, 192))
    else:
        out["prediction_holds"] = None  # a level it needs is missing or not clean
    gates_ok = all(v[0] is True for v in out["gates"].values())
    out["run_valid"] = gates_ok
    out["all_levels_clean"] = all(clean(lv) for lv in out["levels"])

    def mark(v):
        return "ok" if v is True else ("--" if v is None else "FAIL")
    print("\nGATES (-- = evidence missing, counts as not passed)")
    for k, (ok, msg) in sorted(out["gates"].items()):
        print(f"  {k}: {mark(ok)}  {msg}")
    print(f"\nanomaly threshold of G6d (baseline's 99th-percentile relative deviation): {thr}")
    print("LEVELS (RTF counted from the request being sent)")
    print("    N     n  RTFp50 TTFAp50 TTFAp95 undr  audio/s  inflight  G4   G6a  G6b  G6c(Δ)  G6d(share)  G7(p99 ms)")
    f = lambda v, w, p=3: (f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'n/a':>{w}}")
    for lv in out["levels"]:
        print(f"  {lv['level']:<10}{lv['n']:>4} {f(lv['rtf_p50'], 7)} {f(lv['ttfa_p50'], 7)} {f(lv['ttfa_p95'], 7)} "
              f"{lv['requests_with_underrun']:>4} {f(lv['audio_s_per_wall_s'], 8, 1)} {f(lv['mean_in_flight_est'], 8, 1)}  "
              f"{mark(lv['G4_ok']):<5}{mark(lv['G6a_ok']):<5}{mark(lv['G6b_ok']):<5}"
              f"{mark(lv['G6c_ok']):<4}({lv['G6c_preemptions']})  {mark(lv['G6d_ok']):<4}({f(lv['G6d_anomaly_share'], 5, 3).strip()})  "
              f"{mark(lv['G7_ok']):<4}({f(lv['G7_loop_p99_ms'], 6, 1).strip()})")
    print(f"\nrun valid (G0-G3, G5): {gates_ok}; every level clean: {out['all_levels_clean']}")
    print(f"first level with RTF p50 >= 1: {out['first_rtf_ge_1']}; first level with an underrun: {out['first_underrun']}; "
          f"first level failing a per-level gate: {out['first_level_failing_a_G6_or_G7_gate']}")
    print(f"prediction (RTF < 1 at 96 and >= 1 at 128, 160, 192): {out['prediction_holds']}")
    json.dump(out, open(args.result.replace(".json", ".gates.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
