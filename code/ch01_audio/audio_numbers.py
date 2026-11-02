"""Chapter 1: audio as numbers.

    python code/ch01_audio/audio_numbers.py [--wav PATH]

1. Unit arithmetic for the running example (16 kHz mono int16, 160 ms chunks, 10 ms hop).
2. One utterance written as int16 and float32, then read back with the wrong assumptions.
3. Resampling 48 kHz -> 16 kHz properly vs by dropping samples: where a 10 kHz tone goes.
4. The utterance cut into 160 ms chunks (count, samples, bytes, the short last chunk).
Writes results/audio_numbers.json and WAV files under $BOOK_DATA/runs/ch01/.
"""
import argparse
import os

import numpy as np
import soundfile as sf

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, paths, provenance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{paths.BOOK_DATA}/runs/ch01"
DEFAULT_WAV = (f"{paths.BOOK_DATASETS}/librispeech/extracted/LibriSpeech/test-clean/1089/134686/"
               "1089-134686-0006.flac")

ap = argparse.ArgumentParser()
ap.add_argument("--wav", default=DEFAULT_WAV)
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
res = {"provenance": provenance.collect({"input": args.wav})}

# 1. arithmetic -----------------------------------------------------------------
sr = 16000
res["arithmetic"] = {
    "samples_in_5_s_at_16k": audio.samples(5, sr),
    "bytes_per_s_16k_int16_mono": audio.pcm_bytes(1, sr),
    "bytes_per_s_16k_float32_mono": audio.pcm_bytes(1, sr, bits=32),
    "kbps_16k_int16_mono": audio.pcm_bitrate_kbps(sr),
    "kbps_24k_int16_mono": audio.pcm_bitrate_kbps(24000),
    "kbps_48k_int16_stereo": audio.pcm_bitrate_kbps(48000, channels=2),
    "kbps_8k_8bit_mono_(G.711 rate)": audio.pcm_bitrate_kbps(8000, bits=8),
    "samples_per_160ms_chunk_16k": audio.samples(0.160, sr),
    "bytes_per_160ms_chunk_16k_int16": audio.pcm_bytes(0.160, sr),
    "samples_per_10ms_hop_16k": audio.samples(0.010, sr),
    "samples_per_25ms_window_16k": audio.samples(0.025, sr),
    "samples_per_20ms_opus_frame_48k": audio.samples(0.020, 48000),
    "feature_frames_per_160ms_chunk": audio.frames_in(0.160, 10),
    "int16_levels": 2 ** 16,
    "gain_for_half_amplitude_db": float(20 * np.log10(0.5)),
}

# 2. PCM layout -------------------------------------------------------------------
x, file_sr = audio.read_wav(args.wav)
assert file_sr == sr, file_sr
x16 = audio.float_to_int16(x)
sf.write(os.path.join(OUT, "utt_int16.wav"), x16, sr, subtype="PCM_16")
sf.write(os.path.join(OUT, "utt_float32.wav"), x, sr, subtype="FLOAT")
raw = audio.to_le_bytes(x16)
open(os.path.join(OUT, "utt_s16le.raw"), "wb").write(raw)
rows = audio.misread_table(x, sr)
res["pcm"] = {
    "utterance_s": x.size / sr,
    "file_bytes_int16_wav": os.path.getsize(os.path.join(OUT, "utt_int16.wav")),
    "file_bytes_float32_wav": os.path.getsize(os.path.join(OUT, "utt_float32.wav")),
    "raw_s16le_bytes": len(raw),
    "first_4_samples_int16": x16[:4].tolist(),
    "first_8_bytes_hex": raw[:8].hex(" "),
    "level_dbfs": audio.dbfs(x),
    "level_dbfs_after_-6.02_dB": audio.dbfs(audio.gain_db(x, -6.02)),
    "int16_roundtrip_max_abs_error": float(np.max(np.abs(audio.int16_to_float(x16) - x))),
    "misreads": [r.__dict__ for r in rows],
}
be = audio.int16_to_float(audio.from_bytes(raw, ">i2"))
sf.write(os.path.join(OUT, "misread_big_endian.wav"), be, sr, subtype="PCM_16")
sf.write(os.path.join(OUT, "misread_as_stereo_left.wav"), audio.int16_to_float(x16)[0::2], sr, subtype="PCM_16")
sf.write(os.path.join(OUT, "misread_rate_8k.wav"), audio.int16_to_float(x16), sr // 2, subtype="PCM_16")
f32 = audio.from_bytes(raw, "<f4")
res["pcm"]["float32_misread_fraction_nonfinite"] = float(np.mean(~np.isfinite(f32)))
res["pcm"]["float32_misread_fraction_outside_unit"] = float(np.mean(np.abs(f32[np.isfinite(f32)]) > 1))

# 3. resampling -------------------------------------------------------------------
sr48 = 48000
t48 = audio.tone(10000, 1.0, sr48) + audio.tone(1000, 1.0, sr48)
proper = audio.resample(t48, sr48, sr)
naive = audio.naive_decimate(t48, 3)
res["resampling"] = {
    "input": "1 kHz + 10 kHz tones at 48 kHz, 1 s; 10 kHz is above the 8 kHz Nyquist limit of 16 kHz audio",
    "proper_energy_near_6kHz_db": audio.band_energy_db(proper, sr, 5800, 6200),
    "naive_energy_near_6kHz_db": audio.band_energy_db(naive, sr, 5800, 6200),
    "proper_energy_near_1kHz_db": audio.band_energy_db(proper, sr, 900, 1100),
    "naive_energy_near_1kHz_db": audio.band_energy_db(naive, sr, 900, 1100),
    "note": "dropping 2 of every 3 samples folds 10 kHz to 16 - 10 = 6 kHz (an alias); "
            "the windowed-sinc resampler removes it first",
}
up = audio.resample(x, sr, sr48)
back = audio.resample(up, sr48, sr)
res["resampling"]["utt_16k_to_48k_to_16k_max_abs_error"] = float(np.max(np.abs(back[:x.size] - x)))
res["resampling"]["utt_16k_to_48k_to_16k_snr_db"] = float(
    10 * np.log10(np.sum(x.astype(np.float64) ** 2) / np.sum((back[:x.size] - x).astype(np.float64) ** 2)))

# 4. chunking ---------------------------------------------------------------------
cs = list(audio.chunks(x, sr, 160))
res["chunking"] = {
    "chunk_ms": 160,
    "chunk_samples": audio.samples(0.16, sr),
    "n_chunks": len(cs),
    "last_chunk_samples": int(cs[-1].samples.size),
    "last_chunk_ms": cs[-1].samples.size / sr * 1000,
    "first_three": [{"index": c.index, "start_sample": c.start_sample, "end_sample": c.end_sample,
                     "start_ms": c.start_sample / sr * 1000, "bytes_int16": c.samples.size * 2} for c in cs[:3]],
}
provenance.write_json(os.path.join(HERE, "results", "audio_numbers.json"), res)
for k, v in res["arithmetic"].items():
    print(f"{k:40s} {v}")
for r in rows:
    print(f"{r.name:30s} samples={r.n_samples:7d} dur={r.duration_s:6.2f}s peak={r.peak:.3g}")
print({k: v for k, v in res["resampling"].items() if k.endswith("db")})
print(res["chunking"]["n_chunks"], "chunks, last", res["chunking"]["last_chunk_samples"], "samples")
