"""A local voice cascade on one GPU, timed turn by turn (chapter 18).

    microphone (a file played at real time, 32 ms windows)
      -> Silero VAD + endpointer (trailing silence, or Smart Turn v3)       CPU, in this process
      -> streaming ASR over WebSocket (ch15/nemo_server.py, 320 ms chunks)   GPU
      -> LLM, streamed tokens (vLLM OpenAI API, Qwen3-1.7B, no thinking)     GPU
      -> sentence aggregation (common/handoff.py)
      -> streaming TTS (vLLM-Omni Qwen3-TTS, HTTP, raw PCM 24 kHz)           GPU
      -> virtual player (common/player.py)

Per turn, on this process's monotonic clock:
  t_eos         end of user speech: start of streaming + the offline VAD end of speech in the file
  t_endpoint    the endpointer decides the turn is over
  t_final       the ASR's final transcript arrives (after "Done" is sent at t_endpoint)
  t_llm_first   first LLM content token arrives (request sent at t_final)
  t_seg1        the first complete sentence is ready; the first TTS request is sent then
  t_tts_first   the first TTS piece that contains an audible sample arrives (TTFA as in code/tts/hbtts)
  t_play        that first audible sample is heard by the virtual player
voice-to-voice = t_play - t_eos. A turn whose endpoint fires before t_eos is "premature" (the system would
have cut the user off); it is counted and left out of the latency percentiles.

Barge-in (--barge-in): after the agent's first audio is heard, the next turn's audio is fed to the
microphone after --barge-delay s. The moment the VAD declares speech start, playback is stopped, the TTS
and LLM HTTP streams are closed, and the LLM server's /metrics is polled until no request is running.
Echo is not simulated (the agent's audio never reaches the microphone): a real device needs AEC for this.

    python ch18/cascade.py --name cascade_sil500 --endpoint silence --silence-ms 500
    python ch18/cascade.py --name cascade_smart --endpoint smart
    python ch18/cascade.py --name bargein --barge-in --turns 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import websockets

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "tts"))  # the TTS harness of chapters 10-14 (first audible sample)
from common.handoff import SentenceAggregator  # noqa: E402
from common.player import Player  # noqa: E402
from common.turn import WIN, Endpointer, SileroVAD, SmartTurn, speech_end_reference  # noqa: E402
from common.util import RUN_ROOT, RESULTS, error_rate, load_audio, read_utts, run_meta, summarize, write_json  # noqa: E402
from hbtts.audio import first_audible_index  # noqa: E402

TTS_SR = 24000
VOICE = {"en": ("ryan", "English"), "zh": ("vivian", "Chinese")}
SYSTEM = {
    "short": ("You are a friendly voice assistant in a spoken conversation. The user's words come from speech "
              "recognition and may contain errors. Reply in one or two short spoken sentences, in the same "
              "language as the user (English or Simplified Chinese). No lists, no markdown, no emoji."),
    "long": ("You are a friendly voice assistant. Answer in the user's language (English or Simplified Chinese) "
             "with a detailed spoken explanation of at least eight sentences. No lists, no markdown."),
}
MANIFEST = HERE.parent / "data" / "test_utts.jsonl"


def now() -> float:
    return time.monotonic()


class Cascade:
    def __init__(self, a):
        self.a = a
        self.vad = SileroVAD()
        self.smart = SmartTurn() if a.endpoint == "smart" else None
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    # ------------------------------------------------------------------ microphone + VAD + ASR
    async def listen(self, x: np.ndarray, lang: str, rec: dict, t0: float, stop_on: str = "endpoint"):
        """Stream x at real time. Returns when the endpointer fires (stop_on='endpoint') or speech starts
        (stop_on='speech_start'); the ASR session is finalised in the first case."""
        a = self.a
        ep = Endpointer(mode=a.endpoint, silence_ms=a.silence_ms, trigger_ms=a.trigger_ms,
                        fallback_ms=a.fallback_ms, start_ms=a.start_ms, turn_model=self.smart)
        self.vad.reset()
        ws = await websockets.connect(a.asr_url, max_size=None, ping_interval=None) if stop_on == "endpoint" else None
        partials, final = [], asyncio.get_running_loop().create_future()

        async def reader():
            try:
                async for m in ws:
                    d = json.loads(m)
                    d["t"] = now()
                    if d.get("final"):
                        final.set_result(d)
                        return
                    partials.append(d)
            except websockets.exceptions.ConnectionClosed:
                pass
            if not final.done():
                final.set_result(None)

        rt = None
        if ws is not None:
            rt = asyncio.create_task(reader())
            if a.lang_hint:
                await ws.send(json.dumps({"lang": lang}))
        pending, sent = [], 0
        result = None
        n_win = len(x) // WIN
        for i in range(n_win):
            target = t0 + (i + 1) * WIN / 16000
            d = target - now()
            if d > 0:
                await asyncio.sleep(d)
            w = x[i * WIN : (i + 1) * WIN]
            p = self.vad(w)
            if ws is not None:
                pending.append(w)
                if len(pending) == a.asr_windows:
                    await ws.send(np.concatenate(pending).astype("<f4").tobytes())
                    sent += len(pending) * WIN
                    pending = []
            ev = ep.step(p, (i + 1) * WIN / 16000, x[: (i + 1) * WIN] if self.smart else None)
            for e in ev:
                e["wall"] = now()
                rec.setdefault("vad_events", []).append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()})
                if e["type"] == stop_on:
                    result = e
            if result is not None:
                break
        if result is None:
            result = {"type": "forced", "t": n_win * WIN / 16000, "wall": now()}
        if ws is not None:
            if pending:
                await ws.send(np.concatenate(pending).astype("<f4").tobytes())
            await ws.send("Done")
            fin = await asyncio.wait_for(final, timeout=60)
            await ws.close()
            if rt:
                rt.cancel()
            rec["asr_partials"] = len(partials)
            rec["t_final"] = fin["t"] if fin else None
            rec["hyp"] = fin["text"] if fin else ""
        return result

    # ------------------------------------------------------------------ LLM -> aggregation -> TTS
    async def respond(self, text: str, lang: str, rec: dict, player: Player, style: str = "short"):
        a = self.a
        agg = SentenceAggregator(first_min_chars=a.first_min_chars)
        q: asyncio.Queue = asyncio.Queue()
        rec["segments"] = []
        body = {"model": a.llm_model, "stream": True, "temperature": 0.0, "max_tokens": a.max_tokens if style == "short" else a.long_tokens,
                "messages": [{"role": "system", "content": SYSTEM[style]}, {"role": "user", "content": text}],
                "chat_template_kwargs": {"enable_thinking": False}}
        self.llm_resp = None

        async def llm():
            t_req = now()
            rec["t_llm_req"] = t_req
            out = []
            async with self.http.stream("POST", f"{a.llm_url}/v1/chat/completions", json=body) as r:
                self.llm_resp = r
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    d = json.loads(data)
                    delta = d["choices"][0]["delta"].get("content") or ""
                    if not delta:
                        continue
                    t = now()
                    if "t_llm_first" not in rec:
                        rec["t_llm_first"] = t
                    out.append(delta)
                    for seg in agg.feed(delta):
                        await q.put((now(), seg))
            rec["t_llm_done"] = now()
            for seg in agg.flush():
                await q.put((now(), seg))
            await q.put(None)
            rec["llm_text"] = "".join(out)
            rec["llm_chunks"] = len(out)

        async def fetch(seg: str, seg_rec: dict, out: asyncio.Queue):
            """One TTS request; every PCM piece goes to `out` with its arrival time, then None."""
            voice, language = VOICE[lang]
            req = {"model": a.tts_model, "input": seg, "voice": voice, "language": language,
                   "task_type": "CustomVoice", "response_format": "pcm", "stream": True,
                   "stream_format": "audio", "seed": 1234}
            seg_rec["t_req"] = now()
            carry = b""
            try:
                async with self.http.stream("POST", f"{a.tts_url}/v1/audio/speech", json=req) as r:
                    async for b in r.aiter_bytes():
                        t = now()
                        b = carry + b
                        k = len(b) - (len(b) % 2)
                        carry = b[k:]
                        if k:
                            if "t_first_pcm" not in seg_rec:
                                seg_rec["t_first_pcm"] = t
                            await out.put((t, np.frombuffer(b[:k], dtype="<i2").astype(np.float32) / 32768.0))
                seg_rec["t_done"] = now()
            finally:
                await out.put(None)

        async def tts():
            """Sentences are synthesised in order. Without --tts-prefetch the next request is sent when the
            previous one has fully arrived; with it, every sentence is requested as soon as it is ready.
            Either way the player gets the audio in sentence order, each piece at its true arrival time."""
            first_done = False
            n_total = 0
            pcm_all = []
            fetches = []
            pending: asyncio.Queue = asyncio.Queue()

            async def feeder():
                while (item := await q.get()) is not None:
                    t_ready, seg = item
                    seg_rec = {"t_ready": t_ready, "text": seg}
                    rec["segments"].append(seg_rec)
                    out: asyncio.Queue = asyncio.Queue()
                    if a.tts_prefetch:
                        fetches.append(asyncio.create_task(fetch(seg, seg_rec, out)))
                        await pending.put((seg_rec, out, None))
                    else:
                        await pending.put((seg_rec, out, seg))
                await pending.put(None)

            ft = asyncio.create_task(feeder())
            try:
                while (item := await pending.get()) is not None:
                    seg_rec, out, seg = item
                    task = asyncio.create_task(fetch(seg, seg_rec, out)) if seg is not None else None
                    if task:
                        fetches.append(task)
                    while (piece := await out.get()) is not None:
                        t, x = piece
                        player.add(t, len(x))
                        pcm_all.append(x)
                        n_total += len(x)
                        if not first_done:  # on everything received so far, windows aligned to sample 0
                            idx = first_audible_index(np.concatenate(pcm_all), TTS_SR)
                            if idx is not None and idx < n_total:
                                first_done = True
                                rec["t_tts_first"] = t
                                rec["first_audible_sample"] = idx
                    seg_rec["samples"] = n_total
            finally:
                ft.cancel()
                for f in fetches:
                    f.cancel()
                await asyncio.gather(*fetches, return_exceptions=True)  # HTTP streams really closed
                rec["tts_samples"] = n_total
                self.pcm = np.concatenate(pcm_all) if pcm_all else np.zeros(0, np.float32)

        self.llm_task = asyncio.create_task(llm())
        self.tts_task = asyncio.create_task(tts())
        return self.llm_task, self.tts_task

    async def llm_running(self) -> int | None:
        try:
            r = await self.http.get(f"{self.a.llm_url}/metrics", timeout=2.0)
            m = re.search(r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.e+]+)", r.text, re.M)
            return int(float(m.group(1))) if m else None
        except Exception:
            return None

    # ------------------------------------------------------------------ one timed turn
    async def turn(self, utt: dict, x: np.ndarray, ref: tuple[float, float], next_audio=None) -> dict:
        a = self.a
        rec = {"id": utt["id"], "lang": utt["lang"], "ref": utt["text"], "speech_start_s": ref[0], "speech_end_s": ref[1]}
        t0 = now() + 0.05
        rec["t0"] = t0
        rec["t_eos"] = t0 + ref[1]
        ep = await self.listen(x, utt["lang"], rec, t0)
        rec["t_endpoint"] = ep["wall"]
        rec["endpoint_reason"] = ep.get("reason", ep["type"])
        rec["premature"] = bool(ep["t"] < ref[1])
        player = Player(sr=TTS_SR, prebuffer_s=a.prebuffer_ms / 1000)
        self.pcm = None
        llm_t, tts_t = await self.respond(rec["hyp"], utt["lang"], rec, player,
                                          style="long" if next_audio is not None else "short")
        if next_audio is None:
            await asyncio.gather(llm_t, tts_t)
        else:
            await self.barge_in(rec, player, next_audio, llm_t, tts_t)
        if rec.get("first_audible_sample") is not None:
            rec["t_play"] = player.play_time(rec["first_audible_sample"])
        rec["underruns_from_first_pcm"] = player.underruns()
        k = rec.get("first_audible_sample")
        rec["underruns"] = player.underruns(after_sample=k) if k is not None else None
        rec["pieces"] = [[round(t - rec["t0"], 4), n] for t, n in zip(player.arrivals, player.lengths)]
        rec["audio_s"] = round(sum(player.lengths) / TTS_SR, 3)
        if self.pcm is not None and a.save_audio:
            import soundfile as sf

            sf.write(str(self.out / "audio" / f"{utt['id']}.wav"), self.pcm, TTS_SR)
        return rec

    async def barge_in(self, rec, player, nxt, llm_t, tts_t):
        a = self.a
        x2, ref2 = nxt
        while "t_tts_first" not in rec:
            if tts_t.done():
                return
            await asyncio.sleep(0.005)
        t_heard = player.play_time(rec["first_audible_sample"])
        t_start = t_heard + a.barge_delay  # the user starts the interrupting utterance (file start)
        rec["t_barge_audio_start"] = t_start
        rec["t_barge_speech_onset"] = t_start + ref2[0]
        b = {}
        ev = await self.listen(x2, "en", b, t_start, stop_on="speech_start")
        t_det = ev["wall"]
        rec["t_barge_detect"] = t_det
        rec["barge_event"] = ev["type"]
        rec["llm_running_at_detect"] = not llm_t.done()
        rec["tts_running_at_detect"] = not tts_t.done()
        rec["playing_at_detect"] = player.playing_at(t_det)
        player.stop(now())
        rec["t_player_stop"] = player.stopped_at
        for name, task in (("tts", tts_t), ("llm", llm_t)):
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            rec[f"t_{name}_cancelled"] = now()
        # poll the LLM server until it reports no running request (it aborts when the client disconnects)
        t_end = now() + 5
        while now() < t_end:
            n = await self.llm_running()
            if n == 0:
                rec["t_llm_server_idle"] = now()
                break
            await asyncio.sleep(0.01)
        rec["played_s_before_stop"] = round(player.samples_played() / TTS_SR, 3)


def budget(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["premature"] and r.get("t_play") is not None and r.get("t_final")]
    st = lambda f: summarize([f(r) for r in ok], 1000, 0)  # noqa: E731
    seg1 = lambda r: r["segments"][0]["t_ready"]  # noqa: E731
    return {
        "turns_used": len(ok),
        "endpoint_delay_ms": st(lambda r: r["t_endpoint"] - r["t_eos"]),
        "asr_final_after_endpoint_ms": st(lambda r: r["t_final"] - r["t_endpoint"]),
        "llm_request_after_final_ms": st(lambda r: r["t_llm_req"] - r["t_final"]),
        "llm_first_token_ms": st(lambda r: r["t_llm_first"] - r["t_llm_req"]),
        "sentence_aggregation_ms": st(lambda r: seg1(r) - r["t_llm_first"]),
        "tts_first_audio_ms": st(lambda r: r["t_tts_first"] - seg1(r)),
        "playout_ms": st(lambda r: r["t_play"] - r["t_tts_first"]),
        "voice_to_voice_ms": st(lambda r: r["t_play"] - r["t_eos"]),
        "eos_to_final_transcript_ms": st(lambda r: r["t_final"] - r["t_eos"]),
        "underruns_total": sum(r["underruns"] or 0 for r in ok),
        "turns_with_underrun": sum(1 for r in ok if r["underruns"]),
        "turns_with_underrun_from_first_pcm": sum(1 for r in ok if r.get("underruns_from_first_pcm", r["underruns"])),
    }


async def amain(a):
    utts = read_utts(a.manifest)
    en = [u for u in utts if u["lang"] == "en"][: a.turns // 2]
    zh = [u for u in utts if u["lang"] == "zh"][: a.turns - a.turns // 2]
    order = [u for pair in zip(en, zh) for u in pair]
    c = Cascade(a)
    c.out = RUN_ROOT / a.name
    (c.out / "audio").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    # warm-up turn (not recorded): loads kernels and caches in every server
    def prep(u):
        x = load_audio(u["path"])
        tail = (rng.standard_normal(int(a.tail_s * 16000)) * 1e-4).astype(np.float32)
        x = np.concatenate([x, tail])
        return x, speech_end_reference(x, c.vad)
    warm = [u for u in utts if u not in order][0]  # an utterance that is not measured
    wx, wref = prep(warm)
    await c.turn(warm, wx, wref)
    for k, u in enumerate(order):
        x, ref = prep(u)
        if ref is None:
            print(json.dumps({"id": u["id"], "skipped": "no speech found by the offline VAD"}))
            continue
        nxt = None
        if a.barge_in:
            v = order[(k + 1) % len(order)]
            nxt = prep(v)
        rec = await c.turn(u, x, ref, nxt)
        rows.append(rec)
        short = {k2: rec.get(k2) for k2 in ("id", "lang", "premature", "endpoint_reason")}
        if rec.get("t_play"):
            short["v2v_ms"] = round((rec["t_play"] - rec["t_eos"]) * 1000)
        if a.barge_in and rec.get("t_barge_detect"):
            short["detect_ms"] = round((rec["t_barge_detect"] - rec["t_barge_speech_onset"]) * 1000)
            short["llm_idle_ms"] = round((rec.get("t_llm_server_idle", float("nan")) - rec["t_barge_detect"]) * 1000)
        print(json.dumps(short, ensure_ascii=False), flush=True)
        await asyncio.sleep(a.gap_s)
    await c.http.aclose()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--turns", type=int, default=50)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--asr-url", default="ws://127.0.0.1:6010")
    ap.add_argument("--no-lang-hint", dest="lang_hint", action="store_false",
                    help="do not send {'lang': ..} first (needed for ch15/sherpa_server.py)")
    ap.add_argument("--asr-windows", type=int, default=5, help="VAD windows (32 ms) per ASR message: 5 = 160 ms")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--llm-model", default="qwen3-1.7b")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--long-tokens", type=int, default=512)
    ap.add_argument("--tts-url", default="http://127.0.0.1:8091")
    ap.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--endpoint", choices=["silence", "smart"], default="silence")
    ap.add_argument("--silence-ms", type=float, default=500)
    ap.add_argument("--trigger-ms", type=float, default=200)
    ap.add_argument("--fallback-ms", type=float, default=3000)
    ap.add_argument("--start-ms", type=float, default=96)
    ap.add_argument("--first-min-chars", type=int, default=None)
    ap.add_argument("--tts-prefetch", action="store_true", help="request every sentence as soon as it is ready")
    ap.add_argument("--prebuffer-ms", type=float, default=0.0)
    ap.add_argument("--tail-s", type=float, default=3.5, help="noise-floor audio appended after each turn")
    ap.add_argument("--gap-s", type=float, default=0.5)
    ap.add_argument("--barge-in", action="store_true")
    ap.add_argument("--barge-delay", type=float, default=0.5)
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("--results-dir", default=str(RESULTS), help="where ch18_<name>.json goes (default: results/)")
    a = ap.parse_args()
    meta = run_meta(["httpx", "websockets", "onnxruntime", "transformers", "numpy"], {"args": vars(a)})
    rows = asyncio.run(amain(a))
    out = RUN_ROOT / a.name
    with open(out / "turns.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    res = {"meta": meta, "raw": str(out / "turns.jsonl"), "turns": len(rows),
           "premature_endpoints": sum(1 for r in rows if r["premature"]),
           "endpoint_reasons": {k: sum(1 for r in rows if r["endpoint_reason"] == k) for k in sorted({r["endpoint_reason"] for r in rows})},
           "asr_error_rate": {l: error_rate((r["ref"], r["hyp"], l) for r in rows if r["lang"] == l and not r["premature"]) for l in ("en", "zh")},
           "budget_all": budget(rows),
           "budget_by_lang": {l: budget([r for r in rows if r["lang"] == l]) for l in ("en", "zh")}}
    if a.barge_in:
        b = [r for r in rows if r.get("t_barge_detect") and r.get("barge_event") == "speech_start"]
        bp = [r for r in b if r["playing_at_detect"]]
        bl = [r for r in b if r["llm_running_at_detect"]]
        res["barge_in"] = {
            "turns": len(b),
            "llm_running_at_detect": sum(1 for r in b if r["llm_running_at_detect"]),
            "tts_running_at_detect": sum(1 for r in b if r["tts_running_at_detect"]),
            "detect_after_speech_onset_ms": summarize([r["t_barge_detect"] - r["t_barge_speech_onset"] for r in b], 1000, 1),
            "tts_stream_closed_after_detect_ms": summarize([r["t_tts_cancelled"] - r["t_barge_detect"] for r in b], 1000, 2),
            "llm_stream_closed_after_detect_ms": summarize([r["t_llm_cancelled"] - r["t_barge_detect"] for r in bl], 1000, 2),
            "llm_server_idle_after_detect_ms": summarize([r["t_llm_server_idle"] - r["t_barge_detect"] for r in bl if r.get("t_llm_server_idle")], 1000, 1),
            "llm_server_idle_not_seen": sum(1 for r in bl if not r.get("t_llm_server_idle")),
            "speech_onset_to_silence_ms": summarize([r["t_player_stop"] - r["t_barge_speech_onset"] for r in bp], 1000, 1),
            "turns_playing_at_detect": len(bp),
            "forced_no_speech_start": sum(1 for r in rows if r.get("barge_event") not in (None, "speech_start")),
        }
    write_json(out / "summary.json", res)
    write_json(Path(a.results_dir) / f"ch18_{a.name}.json", res)
    print(json.dumps({k: res[k] for k in ("turns", "premature_endpoints", "budget_all")}, ensure_ascii=False, indent=1))
    if "barge_in" in res:
        print(json.dumps(res["barge_in"], indent=1))


if __name__ == "__main__":
    main()
