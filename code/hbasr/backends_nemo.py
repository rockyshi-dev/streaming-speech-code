"""NeMo cache-aware streaming adapter for the latency harness (GPU track, chapters 5, 6 and 9).

NeMo's own example (`CacheAwareStreamingAudioBuffer`) computes the features of the whole file first
and then walks over them; that is fine for accuracy but cannot say *when* a result could have
existed. This adapter is incremental: audio arrives in 160 ms chunks, features are computed only for
frames whose analysis window is fully inside the audio received so far, and an encoder step runs as
soon as the frames it needs exist. Chunk sizes, pre-encode cache and the dropping of extra
pre-encoded frames follow NeMo's `streaming_cfg` and its example loop exactly, so the final text
matches NeMo's own buffer pipeline (checked by ch05_streaming_gpu/check_adapter.py).

Feature timing. NeMo's AudioToMelSpectrogramPreprocessor uses torch.stft with center=True and
n_fft = 512, so feature frame t covers samples [160 t - 256, 160 t + 256). Frame t is usable once
160 t + 256 samples have arrived. Pre-emphasis looks one sample back only, so it is causal.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from . import paths

MODEL_STORE = str(paths.BOOK_MODELS)
NEMO_MODELS = {
    "fastconformer-en-multi": os.path.join(
        MODEL_STORE, "stt_en_fastconformer_hybrid_large_streaming_multi",
        "stt_en_fastconformer_hybrid_large_streaming_multi.nemo"),
}


def nemotron_path() -> str:
    hub = os.path.join(str(paths.hf_home()), "hub",
                       "models--nvidia--nemotron-3.5-asr-streaming-0.6b", "snapshots",
                       "ea30d66debe3740a08b573244286791d423d6b3e", "nemotron-3.5-asr-streaming-0.6b.nemo")
    return hub


def load_model(name: str, device: str = "cuda"):
    import torch
    from nemo.collections.asr.models import ASRModel

    path = nemotron_path() if name == "nemotron-3.5" else NEMO_MODELS[name]
    m = ASRModel.restore_from(path, map_location=torch.device(device))
    m.eval()
    return m


def make_preprocessor(model):
    """The model's own preprocessor with dither 0 and no padding (what NeMo's streaming buffer does)."""
    from omegaconf import OmegaConf

    cfg = copy.deepcopy(model._cfg.preprocessor)
    OmegaConf.set_struct(cfg, False)
    cfg.dither = 0.0
    cfg.pad_to = 0
    pre = model.from_config_dict(cfg)
    return pre.to(model.device).eval()


@dataclass
class NemoConfig:
    model: str = "fastconformer-en-multi"
    att_context_size: tuple = (70, 13)
    decoder: str = "rnnt"            # "rnnt" or "ctc" (hybrid models)
    target_lang: str = ""            # prompt-conditioned models (Nemotron 3.5): e.g. "zh-CN", "en-US", "auto"
    device: str = "cuda"

    def to_dict(self):
        return asdict(self)


class NemoCacheAwareRecognizer:
    sample_rate = 16000
    hop = 160
    half_window = 256  # n_fft // 2

    def __init__(self, cfg: NemoConfig, model=None):
        import torch

        self.torch = torch
        self.cfg = cfg
        self.model = model if model is not None else load_model(cfg.model, cfg.device)
        m = self.model
        if hasattr(m, "change_decoding_strategy") and hasattr(m, "cur_decoder"):
            m.change_decoding_strategy(decoder_type=cfg.decoder, verbose=False)
        if cfg.target_lang and hasattr(m, "set_inference_prompt"):
            m.set_inference_prompt(cfg.target_lang)
        if cfg.att_context_size is not None:
            m.encoder.set_default_att_context_size(list(cfg.att_context_size))
        m.encoder.setup_streaming_params()
        self.scfg = m.encoder.streaming_cfg
        self.pre = make_preprocessor(m)
        self.device = m.device
        pe = m.encoder.pre_encode
        self.sampling_frames = pe.get_sampling_frames() if hasattr(pe, "get_sampling_frames") else None

    # ------------------------------------------------------------ geometry helpers
    def _first(self, v):
        return v[0] if isinstance(v, list) else v

    def _rest(self, v):
        return v[1] if isinstance(v, list) else v

    def describe(self) -> dict:
        s = self.scfg
        return {"chunk_size_frames": s.chunk_size, "shift_size_frames": s.shift_size,
                "pre_encode_cache_size": s.pre_encode_cache_size, "drop_extra_pre_encoded": s.drop_extra_pre_encoded,
                "valid_out_len": s.valid_out_len, "last_channel_cache_size": s.last_channel_cache_size,
                "att_context_size": list(self.model.encoder.att_context_size),
                "subsampling_factor": self.model.encoder.subsampling_factor}

    # ------------------------------------------------------------ protocol
    def start(self) -> None:
        torch = self.torch
        self.audio = np.zeros(0, np.float32)
        self.feats = torch.zeros((1, self.model.encoder._feat_in, 0), device=self.device)
        self.buffer_idx = 0
        self.step = 0
        self.cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        self.prev_hyp = None
        self.pred_out = None
        self.text = ""
        self.finished = False

    def _compute_frames(self, final: bool) -> None:
        """Extend self.feats with every frame that is now complete (all remaining frames if final)."""
        torch = self.torch
        n = self.audio.size
        have = self.feats.shape[-1]
        total = n // self.hop + 1  # frames NeMo produces for the full signal (center=True)
        ready = total if final else max(0, min(total, (n - self.half_window) // self.hop + 1))
        if ready <= have:
            return
        # Recompute only the tail. A sub-signal that starts at s0 = 160 (have - 2) has its frame j
        # centred on absolute sample s0 + 160 j, i.e. absolute frame have - 2 + j. Its frames j >= 2
        # start at least 320 - 256 = 64 samples after s0, so neither the left zero padding nor the
        # missing pre-emphasis predecessor of sample s0 reaches them: they equal the full-signal frames.
        if have >= 2:
            s0, skip = (have - 2) * self.hop, 2
        else:
            s0, skip = 0, have
        seg = self.audio[s0:]
        with torch.inference_mode():
            x = torch.from_numpy(seg).to(self.device)[None]
            f, _ = self.pre(input_signal=x, length=torch.tensor([seg.size], device=self.device))
        new = f[:, :, skip:skip + (ready - have)]
        self.feats = torch.cat([self.feats, new], dim=-1)

    def _run_steps(self, final: bool) -> None:
        torch = self.torch
        s = self.scfg
        nfeat = self.feats.shape[-1]
        while True:
            first = self.buffer_idx == 0
            chunk = self._first(s.chunk_size) if first else self._rest(s.chunk_size)
            shift = self._first(s.shift_size) if first else self._rest(s.shift_size)
            if self.buffer_idx >= nfeat:
                return
            if not final and self.buffer_idx + chunk > nfeat:
                return  # not enough audio yet for a full chunk
            audio_chunk = self.feats[:, :, self.buffer_idx:self.buffer_idx + chunk]
            if self.sampling_frames is not None:  # as NeMo: stop if too short for one output frame
                sf = self.sampling_frames
                need = (sf[0] if first else sf[1]) if isinstance(sf, list) else sf
                if audio_chunk.shape[-1] < need:
                    return
            if first and isinstance(s.pre_encode_cache_size, list):
                cache_pre = torch.zeros((1, audio_chunk.shape[1], s.pre_encode_cache_size[0]), device=self.device)
                zeros = None
            else:
                pcs = self._rest(s.pre_encode_cache_size)
                st = max(0, self.buffer_idx - pcs)
                cache_pre = self.feats[:, :, st:self.buffer_idx]
                zeros = None
                if cache_pre.shape[-1] < pcs:
                    zeros = torch.zeros((1, audio_chunk.shape[1], pcs - cache_pre.shape[-1]), device=self.device)
            added = cache_pre.shape[-1]
            x = torch.cat([cache_pre, audio_chunk], dim=-1)
            if zeros is not None:
                x = torch.cat([zeros, x], dim=-1)
                added += zeros.shape[-1]
            length = torch.tensor([min(x.shape[-1], nfeat - self.buffer_idx + added)], device=self.device)
            self.buffer_idx += shift
            last = final and self.buffer_idx >= nfeat
            with torch.inference_mode():
                (self.pred_out, texts, c0, c1, c2, self.prev_hyp) = self.model.conformer_stream_step(
                    processed_signal=x, processed_signal_length=length,
                    cache_last_channel=self.cache[0], cache_last_time=self.cache[1],
                    cache_last_channel_len=self.cache[2], keep_all_outputs=last,
                    previous_hypotheses=self.prev_hyp, previous_pred_out=self.pred_out,
                    drop_extra_pre_encoded=0 if self.step == 0 else s.drop_extra_pre_encoded,
                    return_transcription=True)
            self.cache = (c0, c1, c2)
            self.step += 1
            t = texts[0]
            self.text = (t.text if hasattr(t, "text") else str(t)) or ""

    def accept(self, x: np.ndarray) -> None:
        self.audio = np.concatenate([self.audio, x.astype(np.float32)])
        self._compute_frames(final=False)
        self._run_steps(final=False)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()

    def partial(self) -> str:
        return self.text.strip()

    def endpoint(self) -> bool:
        return False

    def finish(self) -> str:
        self._compute_frames(final=True)
        self._run_steps(final=True)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        return self.partial()
