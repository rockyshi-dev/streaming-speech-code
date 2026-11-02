"""Chapter 16 additions to the chapter 14 TTS concurrency run: open-loop arrivals and streamed text input.

The chapter 14 harness (code/tts/gpu/qwen3tts_client.py `concurrency`) measures a CLOSED loop: N workers, each
sending its next request when the previous one has fully arrived, so the server never holds more than N
requests. A service does not get to choose that: users arrive when they arrive. This script adds:

  openloop   requests arrive as a Poisson process at `--rates` requests/s (seeded), each sent at its arrival
             time whatever is still in flight; for each rate we report TTFA p50/p95, underruns, and how many
             requests were in flight (sampled at each arrival). HTTP streaming, whole sentence per request.
  wsload     N concurrent WebSocket sessions whose text arrives unit by unit at `--text-rate` units/s
             (a simulated LLM), split_granularity=sentence, i.e. the cascade's text-in/audio-out path, closed loop.

It reuses the chapter 14 client (requests, records, TTFA and underrun definitions) unchanged, so the numbers
are comparable with code/tts/results. Run it in the ssh-tts-gpu environment against a running server:

    TTS_URL=http://127.0.0.1:8091 python ch16/tts_openloop.py openloop --rates 0.5,1,2,4,6,8 --requests 60
    TTS_URL=http://127.0.0.1:8091 python ch16/tts_openloop.py wsload --levels 1,2,4,8,16 --text-rate 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

TTS_GPU = Path(__file__).resolve().parents[2] / "tts" / "gpu"
sys.path.insert(0, str(TTS_GPU))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen3tts_client as q  # noqa: E402  (chapter 14 harness)
from common.util import RESULTS, RUN_ROOT, run_meta, write_json  # noqa: E402


def items():
    it = q.load_items("en", "sent") + q.load_items("zh", "sent")
    h = len(it) // 2
    return [it[i // 2] if i % 2 == 0 else it[h + i // 2] for i in range(len(it))]


async def openloop(a, url):
    order = items()
    rng = np.random.default_rng(a.seed)
    table, rows = [], []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200),
                                     connector=aiohttp.TCPConnector(limit=0)) as s:
        await q.warmup(s, url)
        for rate in [float(x) for x in a.rates.split(",")]:
            gaps = rng.exponential(1.0 / rate, a.requests)
            arrive = np.cumsum(gaps) - gaps[0]
            in_flight = [0]
            seen = []
            level = []

            async def one(k, t_arr, start):
                await asyncio.sleep(max(0.0, start + t_arr - time.perf_counter()))
                seen.append(in_flight[0])
                in_flight[0] += 1
                it = order[k % len(order)]
                rec = await q.req_http(s, url, it["text"], it["lang"])
                in_flight[0] -= 1
                row = q.record_row(rec, it, "http", None, None, f"r{rate}_{k}")
                row.update(rate=rate, arrival_s=float(t_arr), in_flight_at_arrival=seen[-1])
                level.append(row)

            start = time.perf_counter() + 0.1
            await asyncio.gather(*(one(k, t, start) for k, t in enumerate(arrive)))
            wall = time.perf_counter() - start
            rows += level
            ag = q.agg(level)
            table.append({"rate_per_s": rate, **ag, "wall_s": wall,
                          "in_flight_at_arrival": {"p50": float(np.percentile(seen, 50)),
                                                   "p95": float(np.percentile(seen, 95)), "max": int(max(seen))}})
            print(json.dumps({"rate": rate, "n_ok": ag["n_ok"], "ttfa_p50": ag["ttfa_s"].get("p50"),
                              "ttfa_p95": ag["ttfa_s"].get("p95"), "underruns": ag["underruns_total"],
                              "in_flight_p95": table[-1]["in_flight_at_arrival"]["p95"]}), flush=True)
            await asyncio.sleep(3.0)
    return table, rows


async def wsload(a, url):
    order = items()
    table, rows = [], []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200),
                                     connector=aiohttp.TCPConnector(limit=0)) as s:
        await q.warmup(s, url)
        for n in [int(x) for x in a.levels.split(",")]:
            queue: asyncio.Queue = asyncio.Queue()
            for it in order[: a.requests]:
                queue.put_nowait(it)
            level = []

            async def worker(w):
                while True:
                    try:
                        it = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    rec = await q.req_ws(s, url, it["text"], it["lang"], "ws-sentence", a.text_rate)
                    row = q.record_row(rec, it, "ws-sentence", a.text_rate, None, f"c{n}_{it['id']}")
                    row.update(concurrency=n, worker=w)
                    level.append(row)

            t0 = time.perf_counter()
            await asyncio.gather(*(worker(w) for w in range(n)))
            rows += level
            ag = q.agg(level)
            table.append({"concurrency": n, **ag, "wall_s": time.perf_counter() - t0})
            print(json.dumps({"concurrency": n, "n_ok": ag["n_ok"], "ttfa_p50": ag["ttfa_s"].get("p50"),
                              "ttfa_p95": ag["ttfa_s"].get("p95"), "underruns": ag["underruns_total"]}), flush=True)
            await asyncio.sleep(3.0)
    return table, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["openloop", "wsload"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--rates", default="0.5,1,2,4,6,8")
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--text-rate", type=float, default=20.0, help="text units per second (wsload)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    url = os.environ["TTS_URL"]
    name = a.name or f"ch16_{a.mode}"
    t, rows = asyncio.run(openloop(a, url) if a.mode == "openloop" else wsload(a, url))
    out = RUN_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    res = {"meta": run_meta(["vllm", "vllm-omni", "aiohttp", "numpy"],
                            {"args": vars(a), "server": q.server_info(url), "model": q.MODEL, "voices": q.VOICE,
                             "harness": "code/tts/gpu/qwen3tts_client.py (chapter 14), reused unchanged"}),
           "table": t, "raw_dir": str(out)}
    write_json(out / "summary.json", res)
    write_json(RESULTS / f"{name}.json", res)


if __name__ == "__main__":
    main()
