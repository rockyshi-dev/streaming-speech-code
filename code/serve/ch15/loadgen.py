"""Replay N concurrent real-time audio streams against a streaming ASR WebSocket server and time the replies.

Protocol (sherpa-onnx streaming_server.py, extended by ch15/sherpa_server.py and spoken by ch15/nemo_server.py):
client -> server: binary float32 little-endian 16 kHz samples, one message per chunk, then the text "Done".
server -> client: JSON {"text", "n", "final"} where n = number of audio messages consumed when the result was made.

Pacing: chunk i (0-based) is sent at t0 + (i + 1) * chunk, i.e. when its last sample would have left a microphone.
"Done" is sent right after the last chunk.

Two latencies per stream (both client-side, wall clock, on the same machine as the server):
  partial latency  receive time of a non-final result minus the send time of audio message n it reflects
                   = queueing + feature/encoder/decoder compute; it does not include the model's own look-ahead
  final latency    receive time of the final result minus the send time of "Done"

Each of the N workers opens one connection per utterance and walks the manifest from its own offset until the
run's duration is over. Raw per-session records go to <out>/sessions.jsonl; <out>/summary.json holds percentiles.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.util import error_rate, load_audio, read_utts, run_meta, summarize, write_json  # noqa: E402

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "test_utts.jsonl"


def cpu_seconds(pid: int) -> float:
    import os

    f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def slope_ms_per_s(points: list[tuple[float, float]]) -> float | None:
    """Least-squares slope of latency (s) against audio position (s), in ms of latency per s of audio.
    A stream the server keeps up with has a slope near 0; a growing queue gives a positive slope."""
    if len(points) < 3:
        return None
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    if np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0] * 1000.0)


async def run_session(url: str, utt: dict, audio: np.ndarray, chunk_s: float, sr: int, lang_hint: bool) -> dict:
    n_per = int(round(chunk_s * sr))
    chunks = [audio[i : i + n_per] for i in range(0, len(audio), n_per)]
    send_t: dict[int, float] = {}
    msgs: list[dict] = []
    final = asyncio.get_running_loop().create_future()
    rec = {"id": utt["id"], "lang": utt["lang"], "duration": utt["duration"], "chunks": len(chunks)}
    t_connect = time.monotonic()
    async with websockets.connect(url, max_size=None, ping_interval=None, open_timeout=30) as ws:
        rec["connect_s"] = time.monotonic() - t_connect

        async def reader():
            try:
                async for m in ws:
                    t = time.monotonic()
                    d = json.loads(m)
                    d["t"] = t
                    msgs.append(d)
                    if d.get("final"):
                        if not final.done():
                            final.set_result(t)
                        return
            except websockets.exceptions.ConnectionClosed:
                pass
            if not final.done():
                final.set_result(None)

        rt = asyncio.create_task(reader())
        if lang_hint:
            await ws.send(json.dumps({"lang": utt["lang"]}))
        t0 = time.monotonic()
        lags = []
        for i, c in enumerate(chunks):
            target = t0 + (i + 1) * chunk_s
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            await ws.send(c.astype("<f4").tobytes())
            now = time.monotonic()
            send_t[i + 1] = now
            lags.append(now - target)
        t_done = time.monotonic()
        await ws.send("Done")
        t_final = await asyncio.wait_for(final, timeout=120)
        rt.cancel()
    rec["end"] = time.monotonic()
    partial = []
    for d in msgs:
        if d.get("final"):
            continue
        n = int(d["n"])
        if n in send_t:
            partial.append((n * chunk_s, d["t"] - send_t[n]))
    finals = [d for d in msgs if d.get("final")]
    rec.update(
        start=t0,
        partial_latency_s=[round(p[1], 5) for p in partial],
        partial_audio_pos_s=[round(p[0], 3) for p in partial],
        n_messages=len(msgs),
        final_latency_s=None if t_final is None else t_final - t_done,
        send_lag_max_s=max(lags) if lags else 0.0,
        hyp=finals[-1]["text"] if finals else (msgs[-1]["text"] if msgs else ""),
        ref=utt["text"],
        slope_ms_per_s=slope_ms_per_s(partial),
    )
    return rec


async def worker(k: int, args, utts, audios, t_end: float, out: list, rng: random.Random):
    await asyncio.sleep(rng.uniform(0, args.stagger))
    i = (k * 7 + 13 * args.seed) % len(utts)
    while time.monotonic() < t_end:
        u = utts[i % len(utts)]
        try:
            rec = await run_session(args.url, u, audios[u["id"]], args.chunk_ms / 1000.0, 16000, args.lang_hint)
            rec["worker"] = k
            out.append(rec)
        except Exception as e:  # a refused or dropped connection is a result, not a crash
            out.append({"id": u["id"], "worker": k, "error": repr(e)})
            await asyncio.sleep(0.5)
        i += 1


def summarise(sessions: list[dict], warmup_s: float, t_start: float, duration_s: float | None = None) -> dict:
    """Sessions count if they start after the warm-up and end before the run's end (sessions still running
    when the other workers stop see a falling load, so they are left out)."""
    ok = [s for s in sessions if "error" not in s and s.get("start", 0) - t_start >= warmup_s
          and (duration_s is None or s.get("end", 0) <= t_start + duration_s)]
    part = [x for s in ok for x in s["partial_latency_s"]]
    fin = [s["final_latency_s"] for s in ok if s["final_latency_s"] is not None]
    slopes = [s["slope_ms_per_s"] for s in ok if s.get("slope_ms_per_s") is not None]
    by_lang = {}
    for lang in sorted({s["lang"] for s in ok}):
        by_lang[lang] = error_rate((s["ref"], s["hyp"], lang) for s in ok if s["lang"] == lang)
    return {
        "sessions": len(ok),
        "sessions_errored": sum(1 for s in sessions if "error" in s),
        "sessions_dropped_overrun": sum(1 for s in sessions if "error" not in s and duration_s is not None
                                        and s.get("end", 0) > t_start + duration_s),
        "sessions_no_final": sum(1 for s in ok if s["final_latency_s"] is None),
        "audio_s": round(sum(s["duration"] for s in ok), 1),
        "partial_latency_ms": summarize(part, 1000),
        "final_latency_ms": summarize(fin, 1000),
        "send_lag_max_ms": round(max([s["send_lag_max_s"] for s in ok], default=0) * 1000, 1),
        "slope_ms_per_s_median": round(float(np.median(slopes)), 2) if slopes else None,
        "error_rate": by_lang,
    }


async def amain(args):
    utts = read_utts(args.manifest)
    if args.lang:
        utts = [u for u in utts if u["lang"] == args.lang]
    else:  # interleave languages so every worker alternates en / zh
        en = [u for u in utts if u["lang"] == "en"]
        zh = [u for u in utts if u["lang"] == "zh"]
        utts = [u for pair in zip(en, zh) for u in pair] + en[len(zh):] + zh[len(en):]
    audios = {u["id"]: load_audio(u["path"]) for u in utts}
    rng = random.Random(args.seed)
    out: list[dict] = []
    t_start = time.monotonic()
    t_end = t_start + args.duration
    cpu = {}

    async def sample_cpu():  # server CPU time over the measured window only (warm-up .. t_end)
        if not args.server_pid:
            return
        await asyncio.sleep(args.warmup)
        cpu["c0"], cpu["t0"] = cpu_seconds(args.server_pid), time.monotonic()
        await asyncio.sleep(max(0.0, t_end - time.monotonic()))
        cpu["c1"], cpu["t1"] = cpu_seconds(args.server_pid), time.monotonic()

    st = asyncio.create_task(sample_cpu())
    await asyncio.gather(*(worker(k, args, utts, audios, t_end, out, rng) for k in range(args.streams)))
    await st
    return out, t_start, cpu


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:6006")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds during which new sessions start")
    ap.add_argument("--warmup", type=float, default=0.0, help="drop sessions that start in the first W s")
    ap.add_argument("--chunk-ms", type=float, default=160.0)
    ap.add_argument("--stagger", type=float, default=1.0, help="worker start times uniform in [0, stagger] s")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--lang", default=None, help="restrict to en or zh")
    ap.add_argument("--lang-hint", action="store_true", help="send {'lang': ..} first (nemo_server only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--server-pid", type=int, default=0, help="read this process's CPU time over the window")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_meta(["websockets", "numpy"], {"label": args.label, "args": vars(args)})
    sessions, t_start, cpu = asyncio.run(amain(args))
    with open(out_dir / "sessions.jsonl", "w") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    summary = {"meta": meta, "streams": args.streams, "chunk_ms": args.chunk_ms, "t_start": t_start,
               "duration_s": args.duration, **summarise(sessions, args.warmup, t_start, args.duration)}
    if cpu.get("t1"):
        summary["server_cpu_cores"] = round((cpu["c1"] - cpu["c0"]) / (cpu["t1"] - cpu["t0"]), 2)
    summary["meta"]["loadavg_end"] = __import__("os").getloadavg()
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("streams", "sessions", "sessions_errored", "partial_latency_ms",
                                              "final_latency_ms", "slope_ms_per_s_median", "error_rate")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
