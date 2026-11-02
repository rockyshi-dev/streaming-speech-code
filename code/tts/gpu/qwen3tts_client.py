"""Client-side measurements against a running vLLM-Omni Qwen3-TTS server (chapters 12, 14).

Every request is timed on this client's clock (see hbtts.stream for the definitions of
TTFA, RTF and underruns). Three ways of sending text:

  http        POST /v1/audio/speech, stream=true, stream_format=audio, response_format=pcm:
              the whole text is sent at once, audio comes back as raw s16le bytes.
  ws-none     WebSocket /v1/audio/speech/stream, text sent unit by unit (simulated LLM
              tokens) at --rate units/s, split_granularity=none: the server waits for
              input.done, then synthesizes everything.
  ws-sentence same, split_granularity=sentence: each sentence is synthesized as soon as the
              server's splitter sees its end.
  ws-commit   same, text_input_mode=commitment: the server's streaming text frontend
              releases committed segments while text is still arriving.

Sub-commands:
  smoke        one English and one Mandarin request per mode, prints the summaries
  input-modes  chapter 12: paragraphs, all modes, text all at once vs streamed at --rate
  concurrency  chapter 14: N parallel closed-loop streams, N in --levels, http mode

Raw per-request records and audio go to $BOOK_DATA/runs/<name>/,
the summary JSON to code/tts/results/.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hbtts.audio import int16_bytes_to_float  # noqa: E402
from hbtts.runinfo import hf_snapshot, run_dir, runinfo, write_json  # noqa: E402
from hbtts.stream import StreamRecord, summarize  # noqa: E402
from hbtts.textstream import first_sentence_complete_s, schedule, text_units  # noqa: E402

TTS = Path(__file__).resolve().parents[1]
SR = 24000
MODEL = os.environ.get("BOOK_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
VOICE = {"en": ("ryan", "English"), "zh": ("vivian", "Chinese")}
SEED = 1234


def load_items(lang: str, kind: str) -> list[dict]:
    rows = csv.DictReader(open(TTS / "textset" / f"{lang}.tsv", encoding="utf-8"), delimiter="\t")
    return [dict(r, lang=lang) for r in rows if r["kind"] == kind]


class Pcm:
    """Collect s16le bytes that may split a sample across network pieces."""

    def __init__(self, rec: StreamRecord):
        self.rec, self.carry = rec, b""

    def feed(self, t: float, b: bytes) -> None:
        b = self.carry + b
        n = len(b) - (len(b) % 2)
        self.carry = b[n:]
        if n:
            self.rec.add(t, int16_bytes_to_float(b[:n]))


async def req_http(session: aiohttp.ClientSession, url: str, text: str, lang: str) -> StreamRecord:
    voice, language = VOICE[lang]
    body = {"model": MODEL, "input": text, "voice": voice, "language": language,
            "task_type": "CustomVoice", "response_format": "pcm", "stream": True,
            "stream_format": "audio", "seed": SEED}
    rec = StreamRecord(sr=SR, t0=time.perf_counter())
    pcm = Pcm(rec)
    try:
        async with session.post(f"{url}/v1/audio/speech", json=body) as r:
            if r.status != 200:
                rec.error = f"HTTP {r.status}: {(await r.text())[:300]}"
                return rec
            async for piece in r.content.iter_any():
                pcm.feed(time.perf_counter(), piece)
    except Exception as e:  # noqa: BLE001 - record and continue
        rec.error = f"{type(e).__name__}: {e}"
    return rec


async def req_ws(session: aiohttp.ClientSession, url: str, text: str, lang: str, mode: str,
                 rate: float | None) -> StreamRecord:
    voice, language = VOICE[lang]
    cfg = {"type": "session.config", "model": MODEL, "voice": voice, "language": language,
           "task_type": "CustomVoice", "response_format": "pcm", "stream_audio": True, "seed": SEED}
    if mode == "ws-commit":
        cfg["text_input_mode"] = "commitment"
    else:
        cfg["split_granularity"] = {"ws-none": "none", "ws-sentence": "sentence"}[mode]
    plan = schedule(text_units(text), rate)
    rec = StreamRecord(sr=SR)
    pcm = Pcm(rec)
    events = []
    try:
        async with session.ws_connect(url.replace("http", "ws", 1) + "/v1/audio/speech/stream",
                                      max_msg_size=0) as ws:
            await ws.send_json(cfg)

            async def sender():
                start = time.perf_counter()
                rec.t0 = start
                for t_rel, piece in plan:
                    delay = start + t_rel - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    await ws.send_json({"type": "input.text", "text": piece})
                rec.t_text_done = time.perf_counter() - start
                await ws.send_json({"type": "input.done"})

            send_task = asyncio.create_task(sender())
            async for msg in ws:
                now = time.perf_counter()
                if msg.type == aiohttp.WSMsgType.BINARY:
                    pcm.feed(now, msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    ev = json.loads(msg.data)
                    events.append({"t": now - rec.t0 if rec.t0 else None,
                                   **{k: v for k, v in ev.items() if k != "audio_b64"}})
                    if ev.get("type") == "error":
                        rec.error = ev.get("message")
                        break
                    if ev.get("type") == "session.done":
                        break
                else:
                    break
            await send_task
            await ws.send_json({"type": "session.close"})
    except Exception as e:  # noqa: BLE001
        rec.error = f"{type(e).__name__}: {e}"
    rec.events = events  # type: ignore[attr-defined]
    return rec


async def one(session, url, item, mode, rate):
    if mode == "http":
        return await req_http(session, url, item["text"], item["lang"])
    return await req_ws(session, url, item["text"], item["lang"], mode, rate)


def record_row(rec: StreamRecord, item: dict, mode: str, rate, out: Path | None, tag: str) -> dict:
    s = rec.summary()
    row = {"id": item["id"], "lang": item["lang"], "kind": item["kind"], "mode": mode, "rate": rate,
           "text": item["text"], **s,
           "arrivals": rec.arrivals, "piece_samples": [len(p) for p in rec.pieces],
           "events": getattr(rec, "events", None)}
    if rate:
        row["first_sentence_text_complete_s"] = first_sentence_complete_s(item["text"], rate)
    if out is not None and rec.pieces:
        wav = out / "wav" / f"{tag}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav, rec.audio(), SR, subtype="PCM_16")
        row["wav"] = str(wav)
    return row


def agg(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error") and r.get("ttfa_s") is not None]
    return {"n": len(rows), "n_ok": len(ok), "errors": [r["error"] for r in rows if r.get("error")][:5],
            "ttfa_s": summarize([r["ttfa_s"] for r in ok]),
            "first_pcm_s": summarize([r["first_pcm_s"] for r in ok]),
            "rtf": summarize([r["rtf"] for r in ok]),
            "audio_s": summarize([r["audio_s"] for r in ok]),
            "underruns_total": int(sum(r["underruns"] for r in ok)),
            "requests_with_underrun": int(sum(1 for r in ok if r["underruns"] > 0)),
            "stall_s": summarize([r["stall_s"] for r in ok]),
            "inter_arrival_p50_s": summarize([r["inter_arrival_s"]["p50"] for r in ok if "inter_arrival_s" in r]),
            "piece_dur_p50_s": summarize([r["piece_dur_s"]["p50"] for r in ok if "piece_dur_s" in r])}


async def warmup(session, url, n=3):
    items = load_items("en", "sent")[:n] + load_items("zh", "sent")[:n]
    for it in items:
        rec = await req_http(session, url, it["text"], it["lang"])
        if rec.error:
            raise SystemExit(f"warm-up failed: {rec.error}")


def server_info(url: str) -> dict:
    info = {"model": hf_snapshot(MODEL), "url": url}
    try:
        import vllm_omni  # noqa: F401
        pkg = Path(vllm_omni.__file__).parent
        dep = os.environ.get("DEPLOY_CONFIG") or str(pkg / "deploy" / "qwen3_tts.yaml")
        info["deploy_config"] = dep
        info["deploy_config_text"] = Path(dep).read_text()
    except ImportError:
        pass
    return info


async def cmd_smoke(args):
    url = os.environ["TTS_URL"]
    out = run_dir(args.name)
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        await warmup(s, url, 1)
        for lang in ("en", "zh"):
            item = load_items(lang, "para")[0]
            for mode in ("http", "ws-none", "ws-sentence", "ws-commit"):
                rec = await one(s, url, item, mode, args.rate if mode != "http" else None)
                row = record_row(rec, item, mode, args.rate, out, f"{lang}_{mode}")
                rows.append(row)
                print(json.dumps({k: row.get(k) for k in ("lang", "mode", "error", "ttfa_s", "first_pcm_s", "rtf",
                                                           "audio_s", "underruns", "n_pieces", "t_text_done_s",
                                                           "first_sentence_text_complete_s")}))
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def cmd_input_modes(args):
    url = os.environ["TTS_URL"]
    out = run_dir(args.name)
    items = load_items("en", "para") + load_items("zh", "para")
    conds = [("http", None), ("ws-none", None), ("ws-sentence", None), ("ws-commit", None),
             ("ws-none", args.rate), ("ws-sentence", args.rate), ("ws-commit", args.rate)]
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as s:
        await warmup(s, url)
        for rep in range(args.repeats):
            for item in items:
                for mode, rate in conds:
                    rec = await one(s, url, item, mode, rate)
                    tag = f"{item['lang']}_{item['id'].split('+')[0]}_{mode}_r{rate or 0}_rep{rep}"
                    row = record_row(rec, item, mode, rate, out, tag)
                    row["repeat"] = rep
                    rows.append(row)
                    print(json.dumps({k: row.get(k) for k in ("lang", "mode", "rate", "error", "ttfa_s",
                                                               "rtf", "audio_s", "underruns")}), flush=True)
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    table = []
    for mode, rate in conds:
        for lang in ("en", "zh"):
            rs = [r for r in rows if r["mode"] == mode and r["rate"] == rate and r["lang"] == lang]
            fs = [r.get("first_sentence_text_complete_s") for r in rs if r.get("first_sentence_text_complete_s")]
            td = [r["t_text_done_s"] for r in rs if r.get("t_text_done_s") is not None]
            table.append({"mode": mode, "text_rate_units_per_s": rate, "lang": lang, **agg(rs),
                          "first_sentence_text_complete_s": summarize(fs), "text_done_s": summarize(td)})
    summary = {"what": "Qwen3-TTS on vLLM-Omni, one stream at a time: text all at once vs streamed in",
               "runinfo": runinfo(["vllm", "vllm-omni", "torch", "transformers", "aiohttp", "numpy"],
                                  {"server": server_info(url), "repeats": args.repeats,
                                   "text_rate_units_per_s": args.rate, "seed": SEED, "voices": VOICE}),
               "items": [{"id": i["id"], "lang": i["lang"], "text": i["text"]} for i in items],
               "table": table, "raw_dir": str(out)}
    write_json(TTS / "results" / f"{args.name}.json", summary)


async def preemption_count(session, url):
    """Sum of the server's `vllm:num_preemptions` counters (all engines and stages), read from /metrics; None when the
    endpoint is missing or has no such counter. Read-only, so it does not change the load."""
    try:
        async with session.get(url + "/metrics", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return None
            text = await resp.text()
    except Exception:
        return None
    vals = re.findall(r"^vllm:num_preemptions(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", text, re.M)
    return sum(float(v) for v in vals) if vals else None


async def cmd_concurrency(args):
    url = os.environ["TTS_URL"]
    out = run_dir(args.name)
    items = load_items("en", "sent") + load_items("zh", "sent")
    order = [items[i // 2] if i % 2 == 0 else items[len(items) // 2 + i // 2] for i in range(len(items))]
    levels = [int(x) for x in args.levels.split(",")]
    rows = []
    table = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200),
                                     connector=aiohttp.TCPConnector(limit=0)) as s:
        await warmup(s, url)
        # carry-over check: the same sentence five times back to back on one server; a state that
        # leaks from one request into the next (e.g. a growing chunk size) would show up as drift
        b2b = []
        probe = load_items("en", "sent")[0]
        for k in range(5):
            rec = await req_http(s, url, probe["text"], "en")
            sm = rec.summary()
            b2b.append({"k": k, "ttfa_s": sm["ttfa_s"], "first_pcm_s": sm["first_pcm_s"],
                        "first_pieces_samples": [len(p) for p in rec.pieces[:4]], "rtf": sm["rtf"]})
        print(json.dumps({"back_to_back": b2b}), flush=True)
        # event-loop probe: a task that sleeps 10 ms and records how late it wakes. One asyncio process
        # drives all workers and writes a WAV per request, so loop delay would enter TTFA.
        probe_samples: list[tuple[float, float]] = []
        probe_stop = False

        async def loop_probe():
            while not probe_stop:
                t = time.perf_counter()
                await asyncio.sleep(0.01)
                probe_samples.append((t, time.perf_counter() - t - 0.01))

        probe_task = asyncio.ensure_future(loop_probe())
        for visit, n in enumerate(levels):  # a level listed twice is visited twice; `visit` tells them apart
            # untimed burst of n simultaneous requests: the first batch of a new size can pay
            # one-off costs (a batch-4 codec decode stalled 2.8 s the first time it ran)
            preempt_start = await preemption_count(s, url)  # before the warm-up: the burst is part of the level's load
            wres = await asyncio.gather(*(req_http(s, url, order[i % len(order)]["text"], order[i % len(order)]["lang"])
                                          for i in range(n)))
            warmup_errors = sum(1 for r in wres if r.error)
            n_req = max(args.requests, args.min_rounds * n)
            for rep in range(args.repeats):
                queue: asyncio.Queue = asyncio.Queue()
                for i in range(n_req):
                    queue.put_nowait(order[i % len(order)])
                level_rows = []

                async def worker(wid):
                    while True:
                        try:
                            it = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        rec = await req_http(s, url, it["text"], it["lang"])
                        tag = f"v{visit}_c{n}_rep{rep}_w{wid}_{len(level_rows):03d}_{it['lang']}_{it['id']}"
                        row = record_row(rec, it, "http", None, out if args.save_audio else None, tag)
                        row.update(concurrency=n, worker=wid, repeat=rep, visit=visit, t_start_abs=rec.t0)
                        level_rows.append(row)

                wall_start_epoch = time.time()  # local wall clock, to split the server log by level
                t_level = time.perf_counter()
                await asyncio.gather(*(worker(w) for w in range(n)))
                wall = time.perf_counter() - t_level
                wall_end_epoch = time.time()
                preempt_end = await preemption_count(s, url)
                rows.extend(level_rows)
                audio = sum(r["audio_s"] for r in level_rows if not r.get("error"))
                over = sorted(o for (t, o) in probe_samples if t_level <= t <= t_level + wall)
                loop = {"n": len(over), "p50": over[len(over) // 2] if over else None,
                        "p99": over[min(len(over) - 1, int(0.99 * len(over)))] if over else None}
                for lang in ("all", "en", "zh"):
                    rs = [r for r in level_rows if lang == "all" or r["lang"] == lang]
                    in_flight = sum(r["last_audio_s"] for r in rs if r.get("last_audio_s") is not None) / wall
                    table.append({"concurrency": n, "repeat": rep, "visit": visit, "lang": lang, **agg(rs),
                                  "level_wall_s": wall, "audio_s_per_wall_s": audio / wall,
                                  "mean_in_flight_est": in_flight, "loop_overshoot_s": loop,
                                  "preemptions_start": preempt_start, "preemptions_end": preempt_end,
                                  "warmup_errors": warmup_errors,
                                  "wall_start_epoch": wall_start_epoch, "wall_end_epoch": wall_end_epoch})
                t = table[-3]
                print(json.dumps({"concurrency": n, "visit": visit, "rep": rep, "n_ok": t["n_ok"],
                                  "ttfa_p50": t["ttfa_s"].get("p50"), "ttfa_p95": t["ttfa_s"].get("p95"),
                                  "rtf_p50": t["rtf"].get("p50"), "underruns": t["underruns_total"],
                                  "req_with_underrun": t["requests_with_underrun"],
                                  "audio_per_wall": round(audio / wall, 2)}), flush=True)
                await asyncio.sleep(2.0)
        probe_stop = True
        await probe_task
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"what": "Qwen3-TTS on vLLM-Omni: TTFA, RTF and underruns vs concurrent streams, one GPU (runinfo.gpu names it)",
               "runinfo": runinfo(["vllm", "vllm-omni", "torch", "transformers", "aiohttp", "numpy"],
                                  {"server": server_info(url), "levels": levels, "repeats": args.repeats,
                                   "requests_per_level": f"max({args.requests}, {args.min_rounds} x N)",
                                   "per_level_warmup": "one untimed burst of N simultaneous requests", "seed": SEED, "voices": VOICE,
                                   "load_model": "closed loop: N workers, each sends its next request when "
                                                 "the previous one has fully arrived; requests alternate en/zh",
                                   "player": "playback starts at the first audible piece, no pre-buffer"}),
               "back_to_back_check": b2b, "table": table, "raw_dir": str(out)}
    write_json(TTS / "results" / f"{args.name}.json", summary)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("smoke"); a.add_argument("--name", default="qwen3tts_smoke"); a.add_argument("--rate", type=float, default=20.0)
    b = sub.add_parser("input-modes"); b.add_argument("--name", default="ch12_qwen3tts_input_modes")
    b.add_argument("--rate", type=float, default=20.0); b.add_argument("--repeats", type=int, default=2)
    c = sub.add_parser("concurrency"); c.add_argument("--name", default="ch14_qwen3tts_concurrency")
    c.add_argument("--levels", default="1,2,4,8,16"); c.add_argument("--repeats", type=int, default=1)
    c.add_argument("--requests", type=int, default=40); c.add_argument("--save-audio", action="store_true")
    c.add_argument("--min-rounds", type=int, default=3, help="at least this many requests per worker")
    args = ap.parse_args()
    asyncio.run({"smoke": cmd_smoke, "input-modes": cmd_input_modes, "concurrency": cmd_concurrency}[args.cmd](args))


if __name__ == "__main__":
    main()
