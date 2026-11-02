"""CPU stand-ins for the LLM and TTS servers, to dry-run ch18/cascade.py without a GPU (no numbers from
this are reported). LLM: OpenAI-style SSE stream of a fixed reply, one token every 10 ms after 100 ms, with
a vLLM-style /metrics gauge. TTS: /v1/audio/speech streams 24 kHz s16le (a quiet lead-in, then a tone) in
40 ms pieces at 4x real time after 150 ms.

    python ch18/mock_servers.py        # LLM on :8000, TTS on :8091
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
from aiohttp import web

RUNNING = [0]


async def chat(req):
    body = await req.json()
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(req)
    RUNNING[0] += 1
    try:
        await asyncio.sleep(0.1)
        n = 60 if body.get("max_tokens", 80) > 100 else 12
        words = ["Sure", ",", " that", " sounds", " good", "."] + [" More", " words", " here", "."] * n
        for w in words:
            d = {"choices": [{"delta": {"content": w}}]}
            await resp.write(f"data: {json.dumps(d)}\n\n".encode())
            await asyncio.sleep(0.01)
        await resp.write(b"data: [DONE]\n\n")
    finally:
        RUNNING[0] -= 1
    return resp


async def metrics(req):
    return web.Response(text=f'vllm:num_requests_running{{model_name="mock"}} {RUNNING[0]}.0\n')


async def speech(req):
    body = await req.json()
    resp = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
    await resp.prepare(req)
    await asyncio.sleep(0.15)
    dur = 0.08 * len(body["input"])
    t = np.arange(int(dur * 24000)) / 24000
    x = 0.2 * np.sin(2 * np.pi * 220 * t)
    x[:1200] *= 0.001  # 50 ms nearly silent lead-in
    b = (x * 32767).astype("<i2").tobytes()
    step = 960 * 2
    for i in range(0, len(b), step):
        await resp.write(b[i : i + step])
        await asyncio.sleep(0.01)
    return resp


async def main():
    a = web.Application()
    a.router.add_post("/v1/chat/completions", chat)
    a.router.add_get("/metrics", metrics)
    b = web.Application()
    b.router.add_post("/v1/audio/speech", speech)
    import sys

    ports = [int(x) for x in sys.argv[1:3]] or [8000, 8091]
    for app, port in ((a, ports[0]), (b, ports[1])):
        r = web.AppRunner(app)
        await r.setup()
        await web.TCPSite(r, "127.0.0.1", port).start()
    print("READY", flush=True)
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
