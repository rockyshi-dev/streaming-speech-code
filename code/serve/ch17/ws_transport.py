"""Stream "microphone" audio (a file played at real time) over a WebSocket in 20 ms frames, raw PCM or Opus.

    python ch17/ws_transport.py run --codec opus --out $BOOK_DATA/runs/ch17_ws/opus
    python ch17/ws_transport.py run --codec pcm  --out $BOOK_DATA/runs/ch17_ws/pcm
    python ch17/ws_transport.py run --codec opus --jitter wifi --out ...   # SYNTHETIC delay, in-order (TCP)

`run` starts the receiving server as a separate process on 127.0.0.1 and runs the client against it.

Each message is an 12-byte header (uint32 sequence number, float64 capture time on CLOCK_MONOTONIC, shared
by both processes on one machine) followed by the payload: 640 bytes of s16le PCM, or one Opus packet.
The capture time of frame i is the moment its last sample would leave the microphone: t0 + (i + 1) * 20 ms.

Measured per frame (receiver side):
  network_ms   receive time - time the frame was ready to send (loopback, so this is the WebSocket/TCP/asyncio cost, plus any
               synthetic delay the client was told to add)
  capture_to_decoded_ms   receive time + decode time - capture time  (includes the 20 ms frame wait)
  decode_us    Opus decode CPU time
The client also reports encode time and payload bit rate. The receiver writes the decoded audio to the run
directory so it can be listened to.

--jitter adds SYNTHETIC per-frame delays from ch17/jitter.py's scenarios before each send, but keeps
frames in order, which is what a WebSocket over TCP does: one late frame holds back the frames behind it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from common.util import RESULTS, load_audio, read_utts, run_meta, summarize, write_json  # noqa: E402

HDR = struct.Struct("<Id")
SR = 16000
FRAME = 320  # 20 ms at 16 kHz


async def server(port: int, out: Path):
    import opuslib
    import soundfile as sf
    from websockets.asyncio.server import serve

    async def handler(ws):
        dec = None
        rows, pcm = [], []
        async for m in ws:
            t = time.monotonic()
            if isinstance(m, str):
                d = json.loads(m)
                if d.get("type") == "start":
                    dec = opuslib.Decoder(SR, 1) if d["codec"] == "opus" else None
                    continue
                if d.get("type") == "done":
                    audio = np.concatenate(pcm) if pcm else np.zeros(0, np.float32)
                    sf.write(str(out / "received.wav"), audio, SR)
                    await ws.send(json.dumps({"rows": rows}))
                    return
            seq, t_cap = HDR.unpack_from(m)
            payload = m[HDR.size :]
            # after the header the client puts the time the frame was ready to send (captured + encoded)
            t_send = struct.unpack_from("<d", payload)[0]
            payload = payload[8:]
            t0 = time.perf_counter()
            if dec is not None:
                x = np.frombuffer(dec.decode(payload, FRAME), dtype="<i2")
            else:
                x = np.frombuffer(payload, dtype="<i2")
            dt = time.perf_counter() - t0
            pcm.append(x.astype(np.float32) / 32768.0)
            rows.append({"seq": seq, "t_recv": t, "t_send": t_send, "t_cap": t_cap, "decode_s": dt,
                         "bytes": len(payload)})

    async with serve(handler, "127.0.0.1", port, max_size=None, ping_interval=None):
        print("READY", flush=True)
        await asyncio.Future()


async def client(port: int, utt: dict, codec: str, bitrate: int, jitter: str | None, seed: int):
    import opuslib
    from websockets.asyncio.client import connect

    x = load_audio(utt["path"], SR)
    x = np.concatenate([x, np.zeros((-len(x)) % FRAME, np.float32)])
    frames = [x[i : i + FRAME] for i in range(0, len(x), FRAME)]
    enc = None
    if codec == "opus":
        enc = opuslib.Encoder(SR, 1, opuslib.APPLICATION_VOIP)
        enc.bitrate = bitrate
    delays = np.zeros(len(frames))
    if jitter:
        from ch17.jitter import Net, trace

        scen = {"wifi": Net(base_ms=0, jitter_mean_ms=10, spike_p=0.01, spike_ms=120, loss=0.0),
                "mobile": Net(base_ms=0, jitter_mean_ms=25, shape=1.5, spike_p=0.02, spike_ms=250, loss=0.0)}[jitter]
        _, delays, _ = trace(scen, len(frames), seed)
        delays = delays / 1000.0
    enc_t, sizes = [], []
    async with connect(f"ws://127.0.0.1:{port}", max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "start", "codec": codec}))
        t0 = time.monotonic()
        outq: asyncio.Queue = asyncio.Queue()

        async def sender():
            # the (synthetic) network: frame i leaves at its release time; in order, so a late frame holds
            # back the frames behind it (TCP head-of-line blocking). Capture keeps running meanwhile.
            while (item := await outq.get()) is not None:
                release, msg = item
                await asyncio.sleep(max(0.0, release - time.monotonic()))
                await ws.send(msg)

        st = asyncio.create_task(sender())
        release_prev = 0.0
        for i, f in enumerate(frames):
            t_cap = t0 + (i + 1) * FRAME / SR
            s16 = (np.clip(f, -1, 1) * 32767).astype("<i2").tobytes()
            await asyncio.sleep(max(0.0, t_cap - time.monotonic()))  # wait for the frame to be captured
            if enc is not None:
                te = time.perf_counter()
                payload = enc.encode(s16, FRAME)
                enc_t.append(time.perf_counter() - te)
            else:
                payload = s16
            sizes.append(len(payload))
            t_ready = time.monotonic()  # frame captured and encoded: from here on it is "network"
            release = max(release_prev, t_ready + float(delays[i]))
            release_prev = release
            await outq.put((release, HDR.pack(i, t_cap) + struct.pack("<d", t_ready) + payload))
        await outq.put(None)
        await st
        await ws.send(json.dumps({"type": "done"}))
        rows = json.loads(await ws.recv())["rows"]
    return rows, enc_t, sizes, len(frames)


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen([sys.executable, __file__, "server", "--port", str(a.port), "--out", str(out)],
                           stdout=subprocess.PIPE, text=True)
    try:
        assert srv.stdout.readline().strip() == "READY"
        utts = read_utts(Path(__file__).resolve().parents[1] / "data" / "test_utts.jsonl")
        utt = next(u for u in utts if u["id"] == a.utt)
        rows, enc_t, sizes, n = asyncio.run(client(a.port, utt, a.codec, a.bitrate, a.jitter, a.seed))
    finally:
        srv.terminate()
        srv.wait(10)
    net = [r["t_recv"] - r["t_send"] for r in rows]
    c2d = [r["t_recv"] + r["decode_s"] - r["t_cap"] + 0.020 for r in rows]  # + the 20 ms the frame took to fill
    res = {
        "meta": run_meta(["opuslib", "websockets", "numpy"], {"utterance": utt["id"]}),
        "codec": a.codec, "bitrate_target": a.bitrate if a.codec == "opus" else None,
        "jitter_synthetic": a.jitter, "frames": n, "frames_received": len(rows), "frame_ms": 20,
        "payload_kbps": round(8 * float(np.mean(sizes)) / 0.02 / 1000, 1),
        "network_ms": summarize(net, 1000, 2),
        "capture_start_to_decoded_ms": summarize(c2d, 1000, 2),
        "encode_us": summarize(enc_t, 1e6, 1) if enc_t else None,
        "decode_us": summarize([r["decode_s"] for r in rows], 1e6, 1),
        "in_order": all(r["seq"] == i for i, r in enumerate(rows)),
        "notes": ("network_ms is loopback WebSocket time plus any synthetic delay; capture_start_to_decoded_ms "
                  "counts from the first sample of the frame (frame wait 20 ms + network + decode). Opus "
                  "look-ahead is not in it: see ch17_opus_delay.json."),
    }
    write_json(out / "summary.json", res)
    (out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    name = f"ch17_ws_{a.codec}" + (f"_jitter_{a.jitter}" if a.jitter else "")
    write_json(RESULTS / f"{name}.json", res)
    print(json.dumps({k: res[k] for k in ("codec", "jitter_synthetic", "payload_kbps", "network_ms",
                                          "capture_start_to_decoded_ms", "encode_us", "decode_us")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", choices=["run", "server"])
    ap.add_argument("--port", type=int, default=6020)
    ap.add_argument("--out", default=f"{paths.BOOK_DATA}/runs/ch17_ws")
    ap.add_argument("--codec", choices=["pcm", "opus"], default="opus")
    ap.add_argument("--bitrate", type=int, default=24000)
    ap.add_argument("--jitter", choices=["wifi", "mobile"], default=None)
    ap.add_argument("--utt", default="2961-961-0008")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.role == "server":
        asyncio.run(server(a.port, Path(a.out)))
    else:
        run(a)


if __name__ == "__main__":
    main()
