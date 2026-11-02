"""Chapter 2: feed one model features from the wrong front end and watch the WER.

    CUDA_VISIBLE_DEVICES= python code/ch02_features/frontend_mismatch_nemo.py [--n 50]   (ssh-nemo env, CPU)

Model: NVIDIA cache-aware FastConformer hybrid (att_context_size [70,13], full utterance in one
forward pass, transducer head, greedy). Its training front end is NeMo's
AudioToMelSpectrogramPreprocessor: Hann window, 512-point FFT, Slaney mel, log(power + 2^-24),
no normalization ("normalize: NA"). The same N utterances are decoded from:
  nemo            the model's own front end (dither off)
  nemo+utt_cmvn   the same features with per-utterance mean/variance normalization (NeMo "per_feature")
  kaldi_fbank     a Kaldi-style 80-bin log-mel (Povey window, Kaldi mel scale, the sherpa-onnx options)
  kaldi_fbank+offset  Kaldi fbank shifted per bin to the NeMo features' mean over the same utterances
  nemo_int16_scale_samples  the model's features as if samples were int16-scaled (+2 ln 32768 = +20.79)
  nemo_in_dB_not_ln   the model's features in decibels instead of natural log (x 10 / ln 10 = x 4.34)
Each frame sequence is cut to the shorter length. WER uses the en-lower+punct normalizer.
Writes results/frontend_mismatch_nemo.json and per-utterance hypotheses to runs/ch02/.
"""
import argparse
import json
import os

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, data, features, metrics, paths, provenance, textnorm  # noqa: E402
from hbasr.backends_nemo import load_model, make_preprocessor  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{paths.BOOK_DATA}/runs/ch02"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--manifest", default="librispeech-test-clean-200")
args = ap.parse_args()

m = load_model("fastconformer-en-multi", "cpu")
m.encoder.set_default_att_context_size([70, 13])
m.change_decoding_strategy(decoder_type="rnnt", verbose=False)
pre = make_preprocessor(m)
utts = data.read_manifest(args.manifest)[: args.n]


def decode(f):
    with torch.inference_mode():
        f = torch.as_tensor(f, dtype=torch.float32)[None]
        enc, enc_len = m.forward(processed_signal=f, processed_signal_length=torch.tensor([f.shape[-1]]))[:2]
        hyp = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len,
                                                         return_hypotheses=False)
    h = hyp[0][0] if isinstance(hyp, tuple) else hyp[0]
    return (h.text if hasattr(h, "text") else str(h)).strip()


feats = []
for u in utts:
    x, sr = audio.read_wav(u.path)
    with torch.inference_mode():
        nf, nl = pre(input_signal=torch.from_numpy(x)[None], length=torch.tensor([x.size]))
    nf = nf[0, :, : int(nl[0])].numpy()                        # (80, T)
    kf = features.kaldi_fbank(x, sr, snip_edges=False, high_freq=-400.0).T  # (80, T')
    T = min(nf.shape[1], kf.shape[1])
    feats.append((u, nf[:, :T], kf[:, :T]))
offset = np.mean([f[1].mean(1) - f[2].mean(1) for f in feats], axis=0)[:, None]  # per-bin mean offset

variants = {
    "nemo": lambda nf, kf: nf,
    "nemo+utt_cmvn": lambda nf, kf: features.utterance_cmvn(nf.T).T,
    "kaldi_fbank": lambda nf, kf: kf,
    "kaldi_fbank+offset": lambda nf, kf: kf + offset,
    "nemo_int16_scale_samples": lambda nf, kf: nf + 2 * np.log(32768.0),
    "nemo_in_dB_not_ln": lambda nf, kf: nf * 10.0 / np.log(10.0),
}
res = {"provenance": provenance.collect({"manifest": args.manifest, "n": args.n,
                                         "model": "stt_en_fastconformer_hybrid_large_streaming_multi",
                                         "att_context_size": [70, 13], "decoder": "rnnt greedy"}),
       "normalizer": textnorm.EN_DEFAULT.describe(), "wer": {}}
rows = []
for name, fn in variants.items():
    pairs = []
    for u, nf, kf in feats:
        hyp = decode(fn(nf, kf))
        pairs.append((u.text, hyp))
        rows.append({"variant": name, "utt_id": u.utt_id, "ref": u.text, "hyp": hyp})
    ec = metrics.corpus_rate(pairs, textnorm.words, textnorm.EN_DEFAULT)
    res["wer"][name] = ec.to_dict()
    print(name, round(ec.rate * 100, 2), flush=True)
nf, kf = feats[0][1], feats[0][2]
res["feature_stats_first_utt"] = {"nemo_mean": float(nf.mean()), "nemo_std": float(nf.std()),
                                  "kaldi_mean": float(kf.mean()), "kaldi_std": float(kf.std()),
                                  "corr_per_bin_mean": float(np.mean([np.corrcoef(nf[i], kf[i])[0, 1]
                                                                      for i in range(80)]))}
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "frontend_mismatch_nemo.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
res["raw"] = os.path.join(OUT, "frontend_mismatch_nemo.jsonl")
provenance.write_json(os.path.join(HERE, "results", "frontend_mismatch_nemo.json"), res)
