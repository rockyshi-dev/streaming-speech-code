"""Chapter 2: what front end does each exemplar model expect? Read it from the model files.

    python code/ch02_features/model_frontends.py      (ssh-asr env; needs only tar + yaml text)

Extracts the `preprocessor` block of each NeMo checkpoint's model_config.yaml (no model is loaded)
and records sherpa-onnx's front-end defaults (from the installed sherpa_onnx Python signature).
The point to look at is `normalize`: the offline Parakeet CTC model normalizes each utterance with its
own mean and variance ("per_feature"), the streaming models do not ("NA").
Writes results/model_frontends.json.
"""
import inspect
import os
import re
import tarfile

import sherpa_onnx

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import paths, provenance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HF = os.path.join(str(paths.hf_home()), "hub")
NEMO = {
    "stt_en_fastconformer_hybrid_large_streaming_multi (streaming)":
        f"{paths.BOOK_MODELS}/stt_en_fastconformer_hybrid_large_streaming_multi/stt_en_fastconformer_hybrid_large_streaming_multi.nemo",
    "nemotron-3.5-asr-streaming-0.6b (streaming)":
        f"{HF}/models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/ea30d66debe3740a08b573244286791d423d6b3e/nemotron-3.5-asr-streaming-0.6b.nemo",
    "parakeet-ctc-0.6b (offline)":
        f"{HF}/models--nvidia--parakeet-ctc-0.6b/snapshots/ad09ba1cc62743fbc9814de5d2016fca9096485a/parakeet-ctc-0.6b.nemo",
}


def preprocessor_block(path):
    with tarfile.open(path) as t:
        name = next(n for n in t.getnames() if n.endswith("model_config.yaml"))
        text = t.extractfile(name).read().decode()
    m = re.search(r"^preprocessor:\n((?:  .*\n)+)", text, re.M)
    out = {}
    for line in m.group(1).splitlines():
        if re.match(r"^  [A-Za-z_]+:", line):
            k, _, v = line.strip().partition(":")
            out[k] = v.strip()
    return out


res = {"provenance": provenance.collect(), "nemo": {}, "sherpa_onnx_defaults": {}}
for k, p in NEMO.items():
    res["nemo"][k] = {"file": p} | preprocessor_block(p) if os.path.exists(p) else {"file": p, "missing": True}
sig = inspect.signature(sherpa_onnx.OnlineRecognizer.from_transducer)
for k in ("sample_rate", "feature_dim", "low_freq", "high_freq", "dither", "normalize_samples", "snip_edges"):
    res["sherpa_onnx_defaults"][k] = sig.parameters[k].default
res["sherpa_onnx_defaults"]["note"] = ("kaldi-native-fbank: Povey window, 25 ms / 10 ms, pre-emphasis 0.97, "
                                       "DC removal, Kaldi mel scale, natural log; no feature normalization")
provenance.write_json(os.path.join(HERE, "results", "model_frontends.json"), res)
for k, v in res["nemo"].items():
    print(k, {kk: v.get(kk) for kk in ("normalize", "window", "features", "n_fft", "dither")})
print("sherpa-onnx", res["sherpa_onnx_defaults"])
