"""Opus frame sizes, codec delay and loss concealment, measured with libopus through opuslib (CPU).

For each sample rate (16 kHz, the ASR rate, and 48 kHz, the WebRTC rate) and each Opus frame duration
(2.5, 5, 10, 20, 40, 60 ms), one real speech file is encoded and decoded frame by frame, and we record:

  lookahead_ms        encoder look-ahead reported by libopus (OPUS_GET_LOOKAHEAD)
  measured_delay_ms   lag that best aligns decoded with input audio (cross-correlation, 0-40 ms search)
  algorithmic_ms      frame duration + look-ahead: the audio a sender must hold before a frame can leave
  kbps                payload bit rate at the configured target (VBR, 'voip' application)
  enc_us / dec_us     wall time per frame (encode includes the float->int16 conversion), p50 over the file
  snr_db              signal-to-error ratio of the delay-aligned decoded audio against the input

A loss section (20 ms frames, 16 kHz) drops packets at random (synthetic, seeded) and compares the decoder's
packet loss concealment (PLC) with in-band forward error correction (FEC), scoring each against the clean
decode of the same stream.

    python ch17/opus_delay.py --out $BOOK_DATA/runs/ch17_opus
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from common.util import RESULTS, load_audio, read_utts, resample, run_meta, write_json  # noqa: E402

FRAME_MS = [2.5, 5, 10, 20, 40, 60]


def best_lag(ref: np.ndarray, out: np.ndarray, max_lag: int) -> int:
    """Lag L (samples) maximising correlation of out[L:] with ref[:-L]."""
    n = min(len(ref), len(out)) - max_lag
    best, arg = -np.inf, 0
    for lag in range(max_lag + 1):
        c = float(np.dot(ref[:n], out[lag : lag + n]))
        if c > best:
            best, arg = c, lag
    return arg


def snr_db(ref: np.ndarray, out: np.ndarray) -> float:
    n = min(len(ref), len(out))
    e = ref[:n] - out[:n]
    return float(10 * np.log10(np.sum(ref[:n] ** 2) / max(np.sum(e ** 2), 1e-12)))


def to_s16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def from_s16(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def codec_run(x: np.ndarray, fs: int, frame_ms: float, bitrate: int, fec: bool = False, loss_pct: int = 0):
    import opuslib

    enc = opuslib.Encoder(fs, 1, opuslib.APPLICATION_VOIP)
    enc.bitrate = bitrate
    if fec:  # opuslib 3.0.1's property setters for these two are broken; call the ctl directly
        from opuslib.api import ctl
        from opuslib.api.encoder import encoder_ctl

        encoder_ctl(enc.encoder_state, ctl.set_inband_fec, 1)
        encoder_ctl(enc.encoder_state, ctl.set_packet_loss_perc, loss_pct)
    n = int(round(fs * frame_ms / 1000))
    pad = (-len(x)) % n
    xp = np.concatenate([x, np.zeros(pad + 4 * n, np.float32)])  # flush the encoder's look-ahead
    pkts, t_enc = [], []
    for i in range(0, len(xp), n):
        t0 = time.perf_counter()
        pkts.append(enc.encode(to_s16(xp[i : i + n]), n))
        t_enc.append(time.perf_counter() - t0)
    return enc.lookahead, pkts, n, t_enc, xp


def decode(pkts, fs, n, lost=None, use_fec=False):
    import opuslib

    dec = opuslib.Decoder(fs, 1)
    out, t_dec = [], []
    lost = lost or set()
    for i, p in enumerate(pkts):
        t0 = time.perf_counter()
        if i in lost:
            nxt = i + 1
            if use_fec and nxt < len(pkts) and nxt not in lost:
                pcm = dec.decode(pkts[nxt], n, decode_fec=True)  # rebuild frame i from frame i+1's FEC data
            else:
                pcm = dec.decode(b"", n)  # empty packet -> packet loss concealment
        else:
            pcm = dec.decode(p, n)
        t_dec.append(time.perf_counter() - t0)
        out.append(from_s16(pcm))
    return np.concatenate(out), t_dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{paths.BOOK_DATA}/runs/ch17_opus")
    ap.add_argument("--utt", default="2961-961-0008", help="LibriSpeech test-clean id from the manifest")
    ap.add_argument("--bitrate", type=int, default=24000)
    ap.add_argument("--loss", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import opuslib

    utt = next(u for u in read_utts(Path(__file__).resolve().parents[1] / "data" / "test_utts.jsonl") if u["id"] == a.utt)
    x16 = load_audio(utt["path"], 16000)
    rows = []
    for fs in (16000, 48000):
        x = x16 if fs == 16000 else resample(x16, 16000, 48000)
        for fm in FRAME_MS:
            la, pkts, n, t_enc, xp = codec_run(x, fs, fm, a.bitrate)
            y, t_dec = decode(pkts, fs, n)
            lag = best_lag(xp, y, int(0.04 * fs))
            rows.append({
                "fs": fs, "frame_ms": fm, "frame_samples": n,
                "lookahead_samples": la, "lookahead_ms": round(1000 * la / fs, 2),
                "measured_delay_samples": lag, "measured_delay_ms": round(1000 * lag / fs, 2),
                "algorithmic_ms": round(fm + 1000 * la / fs, 2),
                "kbps": round(8 * np.mean([len(p) for p in pkts]) / (fm / 1000) / 1000, 1),
                "bytes_per_packet_mean": round(float(np.mean([len(p) for p in pkts])), 1),
                "enc_us_p50": round(1e6 * float(np.median(t_enc)), 1),
                "dec_us_p50": round(1e6 * float(np.median(t_dec)), 1),
                "snr_db": round(snr_db(xp[: len(y) - lag], y[lag:]), 2),
            })
            print(rows[-1])
    # loss: 16 kHz, 20 ms
    fs, fm = 16000, 20
    rng = np.random.default_rng(a.seed)
    loss_rows = []
    for p in a.loss:
        la, pkts, n, _, xp = codec_run(x16, fs, fm, a.bitrate, fec=True, loss_pct=int(round(p * 100)))
        clean, _ = decode(pkts, fs, n)
        lost = set(int(i) for i in np.nonzero(rng.random(len(pkts)) < p)[0])
        plc, _ = decode(pkts, fs, n, lost, use_fec=False)
        fec, _ = decode(pkts, fs, n, lost, use_fec=True)
        # the FEC-capable encoder spends bits on redundancy; compare with a plain encoder's clean decode too
        _, pk0, _, _, _ = codec_run(x16, fs, fm, a.bitrate)
        loss_rows.append({
            "loss_rate": p, "packets": len(pkts), "lost": len(lost),
            "snr_db_plc_vs_clean": round(snr_db(clean, plc), 2),
            "snr_db_fec_vs_clean": round(snr_db(clean, fec), 2),
            "kbps_with_fec": round(8 * np.mean([len(q) for q in pkts]) / 0.02 / 1000, 1),
            "kbps_without_fec": round(8 * np.mean([len(q) for q in pk0]) / 0.02 / 1000, 1),
        })
        print(loss_rows[-1])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "meta": run_meta(["opuslib", "numpy"], {"libopus": opuslib.api.info.get_version_string().decode(),
                                                 "utterance": utt["id"], "bitrate_target": a.bitrate,
                                                 "application": "voip"}),
        "frames": rows,
        "loss_synthetic": loss_rows,
        "notes": "Loss is synthetic (i.i.d. Bernoulli, seeded). SNR is a waveform measure, not perceived quality.",
    }
    write_json(out / "opus_delay.json", res)
    write_json(RESULTS / "ch17_opus_delay.json", res)


if __name__ == "__main__":
    main()
