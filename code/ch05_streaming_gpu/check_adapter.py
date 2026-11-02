"""Chapters 5 and 9 (GPU track): check that the incremental NeMo adapter computes what NeMo computes.

    python code/ch05_streaming_gpu/check_adapter.py [--n 20]     # one GPU, about 10 minutes

For N utterances of librispeech-test-other-200 and each att_context_size of the multi-latency
FastConformer:
  1. features built incrementally (160 ms at a time, tail recompute) vs the preprocessor run on the
     whole file: max |difference|;
  2. final text of hbasr.backends_nemo (incremental) vs NeMo's own CacheAwareStreamingAudioBuffer
     loop on the whole file (the loop of NeMo's speech_to_text_cache_aware_streaming_infer.py).
Writes results/check_adapter.json.
"""
import argparse
import os

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, data, provenance  # noqa: E402
from hbasr.backends_nemo import NemoCacheAwareRecognizer, NemoConfig, load_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--model", default="fastconformer-en-multi")
ap.add_argument("--manifest", default="librispeech-test-other-200")
ap.add_argument("--target-lang", default="")
args = ap.parse_args()


def nemo_reference(model, x):
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False, pad_and_drop_preencoded=False)
    buf.append_audio(x, stream_id=-1)
    c0, c1, c2 = model.encoder.get_initial_cache_state(batch_size=1)
    prev, pred, text = None, None, ""
    for step, (chunk, lens) in enumerate(iter(buf)):
        with torch.inference_mode():
            pred, texts, c0, c1, c2, prev = model.conformer_stream_step(
                processed_signal=chunk, processed_signal_length=lens, cache_last_channel=c0, cache_last_time=c1,
                cache_last_channel_len=c2, keep_all_outputs=buf.is_buffer_empty(), previous_hypotheses=prev,
                previous_pred_out=pred,
                drop_extra_pre_encoded=0 if step == 0 else model.encoder.streaming_cfg.drop_extra_pre_encoded,
                return_transcription=True)
        t = texts[0]
        text = t.text if hasattr(t, "text") else str(t)
    return text.strip()


model = load_model(args.model)
utts = data.read_manifest(args.manifest)[: args.n]
ctxs = [list(c) for c in model.encoder.att_context_size_all]
res = {"provenance": provenance.collect({"manifest": args.manifest, "n": args.n, "model": args.model}), "per_context": {}}
for ctx in ctxs:
    rec = NemoCacheAwareRecognizer(NemoConfig(model=args.model, att_context_size=tuple(ctx),
                                                target_lang=args.target_lang), model=model)
    same, feat_err, diffs = 0, 0.0, []
    for u in utts:
        x, sr = audio.read_wav(u.path)
        rec.start()
        for c in audio.chunks(x, sr, 160):
            rec.accept(c.samples)
        mine = rec.finish()
        with torch.inference_mode():
            full, _ = rec.pre(input_signal=torch.from_numpy(x).to(model.device)[None],
                              length=torch.tensor([x.size], device=model.device))
        feat_err = max(feat_err, float((full - rec.feats).abs().max()))
        ref = nemo_reference(model, x)
        if mine == ref:
            same += 1
        else:
            diffs.append({"utt_id": u.utt_id, "adapter": mine, "nemo_buffer": ref})
    res["per_context"][str(ctx)] = {"utterances": len(utts), "identical_final_text": same,
                                    "max_abs_feature_diff": feat_err, "differences": diffs,
                                    "streaming_cfg": rec.describe()}
    print(ctx, same, "/", len(utts), "identical; feature max diff", feat_err, flush=True)
provenance.write_json(os.path.join(HERE, "results", f"check_adapter_{args.model}.json"), res)
