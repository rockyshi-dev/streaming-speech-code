"""Chapter 13: decode the same acoustic sequence in chunks and measure the joins.

Part A - a non-causal vocoder (Vocos, mel -> waveform through an inverse STFT).
    The same 100-bin mel spectrogram (hop 256 samples = 10.67 ms at 24 kHz) is decoded
      offline   : in one call (the reference)
      chunked   : in chunks of C mel frames, each decoded on its own, with
                  L frames of left context (decoded, then thrown away),
                  R frames of right context (decoded, thrown away; this is added latency), and
                  an overlap of O frames that is crossfaded (linear) into the next chunk
                  (the overlap frames are also look-ahead, so they add latency too).
Part B - a causal codec decoder (Mimi, from the moshi package), the same codes decoded
      offline          : one call
      stateless chunks : each chunk of C codec frames decoded with fresh state
      streaming        : frame by frame inside mimi.streaming(), state carried across chunks

Measurements per condition, against the offline reference y_ref (error e = y - y_ref):
  boundary_err_db   10*log10( energy of e / energy of y_ref ) in a 5 ms window centred on
                    each chunk join; median and max over the joins
  interior_err_db   the same at the centre of each chunk (control: errors not caused by the join)
  jump_ratio        |y[b] - y[b-1]| at each join / median |y_ref[n] - y_ref[n-1]| over the clip
                    (how much bigger the step across the join is than a typical sample step)
  boundary_level_db 10*log10( energy of y / energy of y_ref ) in the same window: below 0 is a
                    loudness dip at the join (what a crossfade of two out-of-phase signals does)
  snr_db            10*log10( energy of y_ref / energy of e ) over the whole clip
  added_latency_ms  audio that must exist beyond the chunk before it can be emitted (R + O frames)

Run:   python ch13_chunk_boundaries/chunk_boundaries.py
Raw:   $BOOK_DATA/runs/ch13_chunk_boundaries/  (all wavs, rows.jsonl)
Sum.:  code/tts/results/ch13_chunk_boundaries.json, and three-second listening clips in
       code/tts/results/ch13_audio/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hbtts.hub import download, rev  # noqa: E402
from hbtts.runinfo import hf_snapshot, read_prompts, run_dir, runinfo, write_json  # noqa: E402

TTS = Path(__file__).resolve().parents[1]
SR = 24000
HOP = 256
WIN = int(0.005 * SR)  # 5 ms window around a join
RF: dict = {}


def load(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x


def err_db(e: np.ndarray, ref: np.ndarray, centre: int) -> float:
    a, b = max(0, centre - WIN // 2), min(len(ref), centre + WIN // 2)
    pe, pr = float(np.sum(e[a:b] ** 2)), float(np.sum(ref[a:b] ** 2))
    if pr <= 1e-12:
        return float("nan")  # silent region: relative error undefined
    return 10 * np.log10(max(pe, 1e-20) / pr)


def join_metrics(y: np.ndarray, ref: np.ndarray, joins: list[int], chunk: int) -> dict:
    n = min(len(y), len(ref))
    y, ref = y[:n], ref[:n]
    e = y - ref
    joins = [b for b in joins if WIN < b < n - WIN]
    centres = [b - chunk // 2 for b in joins] + ([joins[-1] + chunk // 2] if joins else [])
    centres = [c for c in centres if WIN < c < n - WIN]
    bd = np.array([err_db(e, ref, b) for b in joins])
    lvl = np.array([err_db(y, ref, b) for b in joins])  # level of y relative to ref at the join
    it = np.array([err_db(e, ref, c) for c in centres])
    step = np.median(np.abs(np.diff(ref))) + 1e-12
    jr = np.array([abs(y[b] - y[b - 1]) / step for b in joins])
    snr = 10 * np.log10(np.sum(ref ** 2) / max(np.sum(e ** 2), 1e-20))
    fin = lambda a: a[np.isfinite(a)]  # noqa: E731
    return {"n_joins": len(joins),
            "boundary_err_db_median": float(np.median(fin(bd))) if fin(bd).size else None,
            "boundary_err_db_max": float(np.max(fin(bd))) if fin(bd).size else None,
            "interior_err_db_median": float(np.median(fin(it))) if fin(it).size else None,
            "boundary_level_db_median": float(np.median(fin(lvl))) if fin(lvl).size else None,
            "jump_ratio_median": float(np.median(jr)) if jr.size else None,
            "jump_ratio_max": float(np.max(jr)) if jr.size else None,
            "snr_db": float(snr)}


# ---------------- Part A: Vocos ----------------

def vocos_chunked(vocos, mel: torch.Tensor, C: int, L: int, R: int, O: int) -> np.ndarray:
    """Decode mel (1, 100, T) in chunks of C frames; see the module docstring."""
    T = mel.shape[-1]
    out = np.zeros(T * HOP + 4 * HOP, dtype=np.float32)
    fade = O * HOP
    k = 0
    while k * C < T:
        s, e = k * C, min(T, (k + 1) * C)
        a, b = max(0, s - L), min(T, e + O + R)
        with torch.inference_mode():
            y = vocos.decode(mel[..., a:b])[0].numpy()
        # keep samples of frames [s, e + O): skip left context, drop right context
        keep = y[(s - a) * HOP: (min(T, e + O) - a) * HOP]
        pos = s * HOP
        if k > 0 and fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
            n = min(fade, len(keep))
            out[pos: pos + n] = out[pos: pos + n] * (1 - ramp[:n]) + keep[:n] * ramp[:n]
            out[pos + n: pos + len(keep)] = keep[n:]
        else:
            out[pos: pos + len(keep)] = keep
        k += 1
    return out[: T * HOP]


def receptive_field_frames(vocos, mel: torch.Tensor, rel_tol: float = 1e-3) -> dict:
    """Perturb one mel frame in the middle and see how far, in frames, the output changes."""
    T = mel.shape[-1]
    t0 = T // 2
    m2 = mel.clone()
    m2[..., t0] += 1.0
    with torch.inference_mode():
        d = (vocos.decode(m2) - vocos.decode(mel))[0].abs().numpy()
    idx = np.nonzero(d > rel_tol * d.max())[0]
    return {"perturbed_frame": t0, "rel_tol": rel_tol,
            "frames_before": int(t0 - idx[0] // HOP), "frames_after": int(idx[-1] // HOP - t0)}


def part_a(clips, out: Path, listen: Path) -> list[dict]:
    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz", revision=rev("charactr/vocos-mel-24khz")).eval()
    global RF
    x0 = load(clips[0]["path"])
    with torch.inference_mode():
        RF = receptive_field_frames(vocos, vocos.feature_extractor(torch.from_numpy(x0)[None]))
    print("vocos receptive field", RF, flush=True)
    methods = [  # name, L, R, O
        ("naive", 0, 0, 0),
        ("left_ctx8", 8, 0, 0),
        ("left8_right4", 8, 4, 0),
        ("xfade2", 0, 0, 2),
        ("left8_right2_xfade2", 8, 2, 2),
        ("left16_right8_xfade4", 16, 8, 4),
        ("left32_right32", 32, 32, 0),  # context wider than the measured receptive field (control)
    ]
    rows = []
    for clip in clips:
        x = load(clip["path"])
        with torch.inference_mode():
            mel = vocos.feature_extractor(torch.from_numpy(x)[None])
            ref = vocos.decode(mel)[0].numpy()
        sf.write(out / f"{clip['id']}_vocos_offline.wav", ref, SR, subtype="PCM_16")
        for C in (8, 16, 32):
            for name, L, R, O in methods:
                y = vocos_chunked(vocos, mel, C, L, R, O)
                joins = list(range(C * HOP, len(y), C * HOP))
                m = join_metrics(y, ref, joins, C * HOP)
                rows.append({"part": "A_vocos", "clip": clip["id"], "lang": clip["lang"], "chunk_frames": C,
                             "chunk_ms": 1000 * C * HOP / SR, "method": name, "left_ctx_frames": L,
                             "right_ctx_frames": R, "xfade_frames": O,
                             "added_latency_ms": 1000 * (R + O) * HOP / SR,
                             "compute_frames_per_chunk": C + L + R + O, **m})
                sf.write(out / f"{clip['id']}_vocos_C{C}_{name}.wav", y, SR, subtype="PCM_16")
                if clip is clips[0] and C == 16:
                    sf.write(listen / f"vocos_C16_{name}.wav", y[: 3 * SR], SR, subtype="PCM_16")
                print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()
                                  if k in ("clip", "chunk_frames", "method", "boundary_err_db_median",
                                           "interior_err_db_median", "jump_ratio_median", "snr_db")}), flush=True)
        if clip is clips[0]:
            sf.write(listen / "vocos_offline.wav", ref[: 3 * SR], SR, subtype="PCM_16")
    return rows


# ---------------- Part B: Mimi ----------------

def part_b(clips, out: Path, listen: Path, n_q: int) -> list[dict]:
    from moshi.models import loaders
    mimi = loaders.get_mimi(download(loaders.DEFAULT_REPO, loaders.MIMI_NAME), device="cpu")
    mimi.set_num_codebooks(n_q)
    mimi.eval()
    F = mimi.frame_size  # samples per codec frame
    rows = []
    for clip in clips:
        x = load(clip["path"])
        x = x[: (len(x) // F) * F]
        with torch.inference_mode():
            codes = mimi.encode(torch.from_numpy(x)[None, None])
            ref = mimi.decode(codes)[0, 0].numpy()
        sf.write(out / f"{clip['id']}_mimi_offline.wav", ref, SR, subtype="PCM_16")
        nfr = codes.shape[-1]
        for C in (1, 2, 4):
            for method in ("stateless", "streaming"):
                parts = []
                with torch.inference_mode():
                    if method == "streaming":
                        with mimi.streaming(1):
                            for k in range(0, nfr, C):
                                parts.append(mimi.decode(codes[..., k:k + C])[0, 0].numpy())
                    else:
                        for k in range(0, nfr, C):
                            parts.append(mimi.decode(codes[..., k:k + C])[0, 0].numpy())
                y = np.concatenate(parts)
                joins = list(range(C * F, len(y), C * F))
                m = join_metrics(y, ref, joins, C * F)
                rows.append({"part": "B_mimi", "clip": clip["id"], "lang": clip["lang"], "chunk_frames": C,
                             "chunk_ms": 1000 * C * F / SR, "method": method, "codebooks": n_q,
                             "added_latency_ms": 0.0,
                             "max_abs_diff_vs_offline": float(np.max(np.abs(y[: len(ref)] - ref[: len(y)]))), **m})
                sf.write(out / f"{clip['id']}_mimi_C{C}_{method}.wav", y, SR, subtype="PCM_16")
                if clip is clips[0] and C == 1:
                    sf.write(listen / f"mimi_C1_{method}.wav", y[: 3 * SR], SR, subtype="PCM_16")
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()
                                  if k in ("clip", "chunk_frames", "method", "boundary_err_db_median",
                                           "jump_ratio_median", "snr_db", "max_abs_diff_vs_offline")}), flush=True)
        if clip is clips[0]:
            sf.write(listen / "mimi_offline.wav", ref[: 3 * SR], SR, subtype="PCM_16")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mimi-codebooks", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = run_dir("ch13_chunk_boundaries")
    listen = TTS / "results" / "ch13_audio"
    listen.mkdir(parents=True, exist_ok=True)
    clips = read_prompts(TTS / "textset" / "prompts.tsv")
    rows = part_a(clips, out, listen) + part_b(clips, out, listen, args.mimi_codebooks)
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def mean(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return float(np.mean(v)) if v else None

    table = []
    keys = sorted({(r["part"], r["chunk_frames"], r["method"]) for r in rows},
                  key=lambda t: (t[0], t[1], t[2]))
    for part, C, method in keys:
        rs = [r for r in rows if (r["part"], r["chunk_frames"], r["method"]) == (part, C, method)]
        table.append({"part": part, "chunk_frames": C, "chunk_ms": rs[0]["chunk_ms"], "method": method,
                      "added_latency_ms": rs[0]["added_latency_ms"], "n_clips": len(rs),
                      **{f"{k}_mean_over_clips": mean(rs, k) for k in
                         ("boundary_err_db_median", "boundary_err_db_max", "interior_err_db_median",
                          "boundary_level_db_median",
                          "jump_ratio_median", "jump_ratio_max", "snr_db")}})
    from moshi.models import loaders
    summary = {
        "what": "Chunked decoding vs offline decoding of the same mel / codes: error at the joins",
        "runinfo": runinfo(["torch", "torchaudio", "vocos", "moshi", "numpy"],
                           {"torch_threads": args.threads,
                            "models": [hf_snapshot("charactr/vocos-mel-24khz"),
                                       {**hf_snapshot(loaders.DEFAULT_REPO), "file": loaders.MIMI_NAME}],
                            "mimi_codebooks": args.mimi_codebooks}),
        "clips": clips,
        "definitions": {"window_ms": 1000 * WIN / SR, "hop_samples_vocos": HOP,
                        "mimi_frame_samples": 1920,
                        "note": "dB values are error energy relative to reference energy in the window; "
                                "-inf-like values (perfect match) are floored at 1e-20 energy"},
        "vocos_receptive_field_measured": RF,
        "table": table, "raw_dir": str(out), "listening_clips": str(listen),
    }
    write_json(TTS / "results" / "ch13_chunk_boundaries.json", summary)
    for t in table:
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in t.items()}))


if __name__ == "__main__":
    main()
