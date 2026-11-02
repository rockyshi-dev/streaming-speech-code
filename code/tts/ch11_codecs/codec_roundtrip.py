"""Chapter 10-11 exercise: encode and decode speech with three open neural codecs.

For each codec and number of codebooks this measures, on the CPU:
  * the codec frame rate (codes per second along time), counted from the code tensor
  * codebooks per frame, codebook size, bits per frame, kbit/s and bytes per second
  * reconstruction distance: log-mel L1 between original and decoded audio (dB, lower is closer)
  * encode and decode wall time and their real-time factors
and writes the decoded audio for listening.

It also lists the frame rates of three mel/token front ends for comparison (read from
the model configs; Kokoro's is counted from its predicted durations).

Codecs (transformers implementations, weights from the Hugging Face cache, $HF_HOME):
  kyutai/mimi              (Mimi, 24 kHz)
  facebook/encodec_24khz   (EnCodec, 24 kHz)
  descript/dac_24khz       (DAC, 24 kHz)

Run:
  python ch11_codecs/codec_roundtrip.py --threads 4
Raw output : $BOOK_DATA/runs/ch11_codecs/
Summary    : code/tts/results/ch11_codecs.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hbtts.hub import kokoro_pipeline, kokoro_voice, rev  # noqa: E402
from hbtts.runinfo import hf_snapshot, read_prompts, run_dir, runinfo, write_json  # noqa: E402

TTS = Path(__file__).resolve().parents[1]
SR = 24000


def load_clip(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x


_mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=256, n_mels=80,
                                            f_min=0, f_max=SR / 2, power=1.0)


def logmel_l1_db(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    ma = _mel(torch.from_numpy(a[:n])).clamp_min(1e-5)
    mb = _mel(torch.from_numpy(b[:n])).clamp_min(1e-5)
    return float((20 * (ma.log10() - mb.log10())).abs().mean())


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


class Mimi:
    name, repo = "mimi", "kyutai/mimi"
    settings = [8, 16, 32]
    hop = 1920  # 24,000 / 12.5 (config frame_rate); encoder strides 4*5*6*8 = 960, then 2x downsample

    def __init__(self):
        from transformers import MimiModel
        self.m = MimiModel.from_pretrained(self.repo, revision=rev(self.repo)).eval()
        self.codebook_size = self.m.config.codebook_size

    def encode(self, x, nq):
        return self.m.encode(torch.from_numpy(x)[None, None], num_quantizers=nq).audio_codes

    def decode(self, codes):
        return self.m.decode(codes).audio_values[0, 0].numpy()


class Encodec:
    name, repo = "encodec", "facebook/encodec_24khz"
    settings = [1.5, 6.0, 24.0]  # target bandwidths in kbit/s
    hop = 320  # product of upsampling_ratios 8*5*4*2

    def __init__(self):
        from transformers import EncodecModel
        self.m = EncodecModel.from_pretrained(self.repo, revision=rev(self.repo)).eval()
        self.codebook_size = self.m.config.codebook_size

    def encode(self, x, bw):
        out = self.m.encode(torch.from_numpy(x)[None, None], bandwidth=bw)
        self._scales = out.audio_scales
        return out.audio_codes[0]  # (batch, n_q, frames) for one chunk

    def decode(self, codes):
        return self.m.decode(codes[None], self._scales)[0][0, 0].numpy()


class Dac:
    name, repo = "dac", "descript/dac_24khz"
    settings = [8, 16, 32]
    hop = 320  # product of downsampling_ratios 2*4*5*8 (the config's hop_length field says 512)

    def __init__(self):
        from transformers import DacModel
        self.m = DacModel.from_pretrained(self.repo, revision=rev(self.repo)).eval()
        self.codebook_size = self.m.config.codebook_size

    def encode(self, x, nq):
        return self.m.encode(torch.from_numpy(x)[None, None], n_quantizers=nq).audio_codes

    def decode(self, codes):
        return self.m.decode(audio_codes=codes).audio_values[0].numpy().reshape(-1)


def kokoro_frame_rate() -> dict:
    """Count Kokoro's acoustic frames: audio samples / sum of predicted durations."""
    p = kokoro_pipeline("a", "hexgrad/Kokoro-82M")
    rows = []
    for r in p("After early nightfall the yellow lamps would light up, here and there.",
               voice=kokoro_voice("hexgrad/Kokoro-82M", "af_heart")):
        rows.append((int(r.pred_dur.sum()), len(r.audio)))
    frames = sum(f for f, _ in rows)
    samples = sum(s for _, s in rows)
    return {"frames": frames, "samples": samples, "samples_per_frame": samples / frames,
            "frame_rate_hz": SR * frames / samples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prompts", default=str(TTS / "textset" / "prompts.tsv"))
    ap.add_argument("--summary", default=str(TTS / "results" / "ch11_codecs.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = run_dir("ch11_codecs")
    clips = read_prompts(args.prompts)

    rows = []
    for cls in (Mimi, Encodec, Dac):
        codec = cls()
        for clip in clips:
            x = load_clip(clip["path"])
            dur = len(x) / SR
            sf.write(out / f"{clip['id']}_orig24k.wav", x, SR)
            for setting in cls.settings:
                with torch.inference_mode():
                    codec.decode(codec.encode(x[: SR // 2], setting))  # warm-up on 0.5 s
                    codes, t_enc = timed(lambda: codec.encode(x, setting))
                    y, t_dec = timed(lambda: codec.decode(codes))
                n_q, n_frames = int(codes.shape[-2]), int(codes.shape[-1])
                bits = n_q * math.log2(codec.codebook_size)
                fr = n_frames / dur
                name = f"{clip['id']}_{codec.name}_{setting}"
                sf.write(out / f"{name}.wav", y, SR)
                rows.append({
                    "codec": codec.name, "setting": setting, "clip": clip["id"], "lang": clip["lang"],
                    "clip_s": dur, "n_frames": n_frames, "codebooks": n_q,
                    "codebook_size": codec.codebook_size, "frame_rate_hz_counted": fr,
                    "bits_per_frame": bits, "kbit_per_s": fr * bits / 1000.0,
                    "bytes_per_s": fr * bits / 8.0,
                    "logmel_l1_db": logmel_l1_db(x, y), "out_samples": len(y),
                    "encode_s": t_enc, "decode_s": t_dec, "encode_rtf": t_enc / dur, "decode_rtf": t_dec / dur,
                })
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}))
        del codec

    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # one line per codec setting, averaged over the clips
    table = []
    for codec in ("mimi", "encodec", "dac"):
        for setting in sorted({r["setting"] for r in rows if r["codec"] == codec}):
            rs = [r for r in rows if r["codec"] == codec and r["setting"] == setting]
            hop = {"mimi": Mimi.hop, "encodec": Encodec.hop, "dac": Dac.hop}[codec]
            fr = SR / hop
            table.append({
                "codec": codec, "setting": setting, "codebooks": rs[0]["codebooks"],
                "codebook_size": rs[0]["codebook_size"],
                "samples_per_frame": hop, "frame_rate_hz": fr, "frame_ms": 1000.0 * hop / SR,
                "frame_rate_hz_counted": float(np.mean([r["frame_rate_hz_counted"] for r in rs])),
                "frames_minus_duration_over_hop": [r["n_frames"] - r["clip_s"] * SR / hop for r in rs],
                "bits_per_frame": rs[0]["bits_per_frame"],
                "kbit_per_s": fr * rs[0]["bits_per_frame"] / 1000.0,
                "bytes_per_s": fr * rs[0]["bits_per_frame"] / 8.0,
                "lm_steps_per_s_flattened": fr * rs[0]["codebooks"],
                "ms_per_step_flattened": 1000.0 / (fr * rs[0]["codebooks"]),
                "ms_per_step_depth_transformer": 1000.0 / fr,
                "logmel_l1_db_mean": float(np.mean([r["logmel_l1_db"] for r in rs])),
                "encode_rtf_mean": float(np.mean([r["encode_rtf"] for r in rs])),
                "decode_rtf_mean": float(np.mean([r["decode_rtf"] for r in rs])),
            })

    qcfg = json.load(open(next(Path(hf_snapshot_path("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")).glob(
        "speech_tokenizer/config.json"))))
    front_ends = {
        "vocos_mel_24khz": {"source": "charactr/vocos-mel-24khz config.yaml (hop_length 256, n_mels 100)",
                            "frame_rate_hz": SR / 256, "samples_per_frame": 256},
        "kokoro_82m_counted": kokoro_frame_rate() | {"source": "hexgrad/Kokoro-82M, counted from pred_dur"},
        "qwen3_tts_12hz_tokenizer": {
            "source": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice speech_tokenizer/config.json",
            "samples_per_frame": qcfg["decode_upsample_rate"],
            "frame_rate_hz": qcfg["output_sample_rate"] / qcfg["decode_upsample_rate"],
            "num_quantizers": qcfg["decoder_config"]["num_quantizers"],
            "codebook_size": qcfg["decoder_config"]["codebook_size"],
            "decoder_sliding_window_frames": qcfg["decoder_config"]["sliding_window"]},
    }

    summary = {
        "what": "Codec frame rate, codebooks, bitrate and reconstruction distance, CPU",
        "runinfo": runinfo(["torch", "torchaudio", "transformers", "kokoro", "numpy"],
                           {"torch_threads": args.threads,
                            "models": [hf_snapshot(c.repo) for c in (Mimi, Encodec, Dac)]
                            + [hf_snapshot("hexgrad/Kokoro-82M"),
                               hf_snapshot("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")]}),
        "clips": clips,
        "metric_notes": {
            "logmel_l1_db": "mean |20 log10 mel_orig - 20 log10 mel_decoded| over 80 mel bins, "
                            "n_fft 1024, hop 256, magnitude mel floored at 1e-5",
            "settings": "mimi/dac: number of codebooks; encodec: target bandwidth in kbit/s",
            "frame_rate": "frame_rate_hz = 24,000 / samples_per_frame; the counted rate (frames / clip "
                          "duration) differs from it by the last partial frame: Mimi and EnCodec pad it to a whole "
                          "frame, DAC drops it",
            "timing": "one warm-up encode+decode on 0.5 s per clip and setting, then one timed pass",
            "aishell_clips": "16 kHz originals resampled to 24 kHz, so there is no content above 8 kHz",
        },
        "table": table,
        "front_end_frame_rates": front_ends,
        "raw_dir": str(out),
    }
    write_json(Path(args.summary), summary)
    print(json.dumps(table, indent=1))


def hf_snapshot_path(repo: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=rev(repo), allow_patterns=["speech_tokenizer/config.json"])


if __name__ == "__main__":
    main()
