"""A GPU streaming ASR server: NeMo cache-aware FastConformer-RNNT (Nemotron 3.5 ASR streaming 0.6B), many
streams batched into one encoder step.

What it shows (chapter 15):
  * the model as a step function: (feature chunk, caches, decoder state) in -> (tokens, new caches) out;
  * per-stream state kept per WebSocket session: incremental features, attention and convolution caches,
    the transducer hypothesis (decoder state + tokens), and the language prompt; freed on disconnect;
  * a batched step loop: every stream whose next chunk is complete is gathered into one batch (caches
    concatenated along the batch dimension), stepped once on the GPU, and scattered back.

Protocol: the same as ch15/sherpa_server.py (binary float32 16 kHz chunks, then "Done"; replies
{"text", "n", "final"}), plus an optional first text message {"lang": "en"|"zh"} that selects the
language prompt (default "auto").

Scheduling policy: no waiting to fill a batch. As soon as the GPU is free, every ready stream (up to
--max-batch) goes into the next step. Streams at their first chunk, and streams finishing, have different
chunk shapes and are stepped as separate groups.

    python ch15/nemo_server.py --port 6010 --context-right 3          # 320 ms chunks (trained: r = 0, 3, 6, 13)
    python ch15/nemo_server.py --selftest 8                            # compare with NeMo's own streaming loop
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import re
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from common.stream_features import StreamFeaturizer  # noqa: E402

DEFAULT_MODEL = str(paths.hf_home() / "hub" / "models--nvidia--nemotron-3.5-asr-streaming-0.6b" / "snapshots" /
                    "ea30d66debe3740a08b573244286791d423d6b3e" / "nemotron-3.5-asr-streaming-0.6b.nemo")
LANG_PROMPT = {"en": "en-US", "zh": "zh-CN", "auto": "auto"}


_TAG = re.compile(r"\s*<[a-z]{2,3}(-[A-Za-z]{2,4})?>")


def clean_text(t: str) -> str:
    """Drop the language tags (e.g. '<en-US>') Nemotron emits at sentence ends."""
    return _TAG.sub("", t).strip()


def _row_prompt(self, encoded):
    """Per-row language prompt (NeMo's PromptStreamingMixin uses one prompt for the whole batch)."""
    idx = getattr(self, "_row_prompt_idx", None)
    if not getattr(self, "concat", False) or idx is None:
        return encoded
    enc = encoded.transpose(1, 2)
    b, t, _ = enc.shape
    prompt = torch.zeros(b, t, self.num_prompts, dtype=enc.dtype, device=enc.device)
    prompt.scatter_(2, idx.view(b, 1, 1).expand(-1, t, -1).to(enc.device), 1.0)
    out = self.prompt_kernel(torch.cat([enc, prompt], dim=-1)).to(enc.dtype)
    return out.transpose(1, 2)


class Session:
    _next = 0

    def __init__(self, engine: "BatchedStreamingASR", lang: str = "auto"):
        Session._next += 1
        self.sid = Session._next
        self.eng = engine
        self.lang = lang
        self.prompt_idx = engine.prompt_index(lang)
        self.feat = StreamFeaturizer(engine.mel_cpu, hop=160, half_window=256, count_last_frame=False)
        self._buf = np.zeros((1024, engine.n_mels), np.float32)  # feature frames, grown by doubling
        self.n_frames = 0
        self.frames_after_msg: list[int] = []   # frames available after audio message m (1-based index m-1)
        self.idx = 0                            # first feature frame of the next chunk
        self.step = 0
        self.cache = engine.initial_cache()
        self.hyp = None
        self.text = ""
        self.done = False
        self.finished = False

    # ---- input side (event loop thread)
    def accept(self, samples: np.ndarray) -> None:
        self._append(self.feat.accept(samples))
        self.frames_after_msg.append(self.n_frames)

    def input_done(self) -> None:
        self._append(self.feat.finish())
        self.done = True

    def _append(self, f: np.ndarray) -> None:
        if not f.size:
            return
        need = self.n_frames + len(f)
        if need > len(self._buf):
            nb = np.zeros((max(need, 2 * len(self._buf)), self._buf.shape[1]), np.float32)
            nb[: self.n_frames] = self._buf[: self.n_frames]
            self._buf = nb
        self._buf[self.n_frames : need] = f
        self.n_frames = need

    def feats(self) -> np.ndarray:
        return self._buf[: self.n_frames]

    # ---- scheduling
    def need(self) -> int:
        """Frames the next regular step needs to have arrived."""
        return self.idx + (self.eng.chunk[0] if self.step == 0 else self.eng.chunk[1])

    def kind(self) -> str | None:
        """'first' / 'next' / 'final' if a step can run now, else None."""
        if self.finished:
            return None
        if self.n_frames >= self.need() and not (self.done and self.n_frames == self.need()):
            return "first" if self.step == 0 else "next"
        if self.done:
            return "final"
        return None

    def msg_for(self, frame_end: int) -> int:
        """1-based index of the audio message that made frames [0, frame_end) available."""
        for m, f in enumerate(self.frames_after_msg, start=1):
            if f >= frame_end:
                return m
        return len(self.frames_after_msg)


class BatchedStreamingASR:
    def __init__(self, model_path: str, context_right: int, device: str = "cuda", dtype: str = "fp32",
                 max_batch: int = 64, allow_untrained: bool = False):
        import nemo.collections.asr as nemo_asr
        from omegaconf import open_dict

        self.device = device
        m = nemo_asr.models.ASRModel.restore_from(model_path, map_location="cpu")
        m.preprocessor.featurizer.dither = 0.0
        m.preprocessor.featurizer.pad_to = 0
        trained = [list(x) for x in m.cfg.encoder.get("att_context_size", [])]
        if trained and isinstance(trained[0], list) and [56, context_right] not in trained and not allow_untrained:
            raise ValueError(f"att_context_size [56, {context_right}] is not one the model was trained with: {trained}")
        self.trained_contexts = trained
        m.encoder.set_default_att_context_size([56, context_right])
        m.encoder.setup_streaming_params()
        dc = m.cfg.decoding
        with open_dict(dc):
            dc.strategy = "greedy_batch"
            dc.greedy.max_symbols = 10
            dc.fused_batch_size = -1
        m.change_decoding_strategy(dc, verbose=False)
        # a second front end on the CPU, built from the same config (it has no learned weights)
        self.pre_cpu = type(m).from_config_dict(m.cfg.preprocessor).cpu().eval()
        self.pre_cpu.featurizer.dither = 0.0
        self.pre_cpu.featurizer.pad_to = 0
        m = m.to(device).eval()
        m._apply_prompt_to_encoded = types.MethodType(_row_prompt, m)
        self.model = m
        cfg = m.encoder.streaming_cfg
        self.cfg = cfg
        self.chunk = cfg.chunk_size if isinstance(cfg.chunk_size, list) else [cfg.chunk_size] * 2
        pec = cfg.pre_encode_cache_size
        self.pre_cache = pec[1] if isinstance(pec, list) else pec
        self.drop_extra = cfg.drop_extra_pre_encoded
        sf = m.encoder.pre_encode.get_sampling_frames()
        self.min_frames = sf if isinstance(sf, list) else [sf, sf]
        self.n_mels = m.cfg.preprocessor.features
        self.max_batch = max_batch
        self.amp = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
        self.prompts = dict(m.cfg.model_defaults.get("prompt_dictionary", {}))
        self.chunk_ms = (context_right + 1) * 80
        torch.set_num_threads(1)  # the CPU front end runs on tiny segments; a thread pool only adds spinning

    def prompt_index(self, lang: str) -> int:
        return int(self.prompts.get(LANG_PROMPT.get(lang, lang), self.prompts.get("auto", 0)))

    @torch.inference_mode()
    def mel_cpu(self, seg: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32))[None]
        f, n = self.pre_cpu(input_signal=x, length=torch.tensor([x.shape[1]]))
        return f[0].T.numpy()  # all frames; StreamFeaturizer never uses the masked last one

    def initial_cache(self):
        c = self.model.encoder.get_initial_cache_state(batch_size=1)
        return tuple(t.to(self.device) for t in c)

    def new_session(self, lang: str = "auto") -> Session:
        return Session(self, lang)

    def ready_groups(self, sessions) -> list[tuple[str, list[Session]]]:
        groups: dict[str, list[Session]] = {}
        for s in sessions:
            k = s.kind()
            if k == "final" and s.step == 0:
                k = "final0"
            if k:
                groups.setdefault(k, []).append(s)
        out = []
        for k in ("final", "final0", "next", "first"):
            g = groups.get(k, [])
            for i in range(0, len(g), self.max_batch):
                out.append((k, g[i : i + self.max_batch]))
        return out

    def _inputs(self, kind: str, s: Session):
        f = s.feats()
        if kind in ("first", "final0"):
            end = min(s.n_frames, s.idx + self.chunk[0]) if kind == "first" else s.n_frames
            x = f[s.idx : end]
            return x, end
        end = s.n_frames if kind == "final" else s.idx + self.chunk[1]
        lo = max(0, s.idx - self.pre_cache)
        x = f[lo:end]
        pad = self.pre_cache - (s.idx - lo)
        if pad > 0:
            x = np.concatenate([np.zeros((pad, f.shape[1]), np.float32), x], axis=0)
        return x, end

    @torch.inference_mode()
    def run_group(self, kind: str, group: list[Session]) -> list[tuple[Session, str, bool, int]]:
        """One batched encoder + decoder step. Returns (session, text, is_final, audio message index)."""
        final = kind in ("final", "final0")
        xs, ends, keep = [], [], []
        for s in group:
            x, end = self._inputs(kind, s)
            new = end - s.idx
            need_min = self.min_frames[0] if kind in ("first", "final0") else self.min_frames[1]
            # NeMo's offline front end leaves one masked, zero frame after the last valid one, and its streaming
            # loop counts it when deciding whether a last chunk is long enough; do the same so the two agree.
            if final:
                x = np.concatenate([x, np.zeros((1, x.shape[1]), np.float32)], axis=0)
            if final and new + 1 < need_min:
                keep.append(False)  # too short to produce an output frame; finish with the text so far
            else:
                keep.append(True)
            xs.append(x)
            ends.append(end)
        run = [i for i, k in enumerate(keep) if k]
        results = []
        if run:
            sub = [group[i] for i in run]
            lens = torch.tensor([xs[i].shape[0] - (1 if final else 0) for i in run], dtype=torch.long)
            T = max(xs[i].shape[0] for i in run)
            batch = np.zeros((len(run), T, self.n_mels), np.float32)
            for j, i in enumerate(run):
                batch[j, : xs[i].shape[0]] = xs[i]
            sig = torch.from_numpy(batch).transpose(1, 2).to(self.device)
            c0 = torch.cat([s.cache[0] for s in sub], dim=1)
            c1 = torch.cat([s.cache[1] for s in sub], dim=1)
            c2 = torch.cat([s.cache[2] for s in sub], dim=0)
            self.model._row_prompt_idx = torch.tensor([s.prompt_idx for s in sub], dtype=torch.long)
            hyps = None if kind in ("first", "final0") else [s.hyp for s in sub]
            drop = 0 if kind in ("first", "final0") else self.drop_extra
            ctx = torch.autocast("cuda", dtype=self.amp) if self.amp is not None else torch.autocast("cuda", enabled=False)
            with ctx:
                out = self.model.conformer_stream_step(
                    processed_signal=sig, processed_signal_length=lens.to(self.device),
                    cache_last_channel=c0, cache_last_time=c1, cache_last_channel_len=c2,
                    keep_all_outputs=final, previous_hypotheses=hyps, drop_extra_pre_encoded=drop,
                    return_transcription=True)
            _, _, n0, n1, n2, best = out[:6]
            for j, s in enumerate(sub):
                s.cache = (n0[:, j : j + 1], n1[:, j : j + 1], n2[j : j + 1])
                s.hyp = best[j]
                t = best[j].text
                t = t if isinstance(t, str) else self.model.tokenizer.ids_to_text(list(best[j].y_sequence))
                s.text = clean_text(t)
        for i, s in enumerate(group):
            n_msg = s.msg_for(ends[i])
            s.idx = ends[i]
            s.step += 1
            if final:
                s.finished = True
            results.append((s, s.text, final, n_msg))
        return results


# ------------------------------------------------------------------ WebSocket server

class Server:
    def __init__(self, eng: BatchedStreamingASR, step_log: str | None):
        self.eng = eng
        self.sessions: dict[int, tuple[Session, object]] = {}
        self.wake = asyncio.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self.step_log = open(step_log, "a") if step_log else None

    async def handler(self, ws):
        sess = None
        final_ev = asyncio.Event()
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    if msg == "Done":
                        if sess is None:
                            sess = self._add(ws, "auto", final_ev)
                        sess.input_done()
                        self.wake.set()
                        break
                    d = json.loads(msg)
                    if sess is None:
                        sess = self._add(ws, d.get("lang", "auto"), final_ev)
                    continue
                if sess is None:
                    sess = self._add(ws, "auto", final_ev)
                sess.accept(np.frombuffer(msg, dtype="<f4"))
                if sess.kind():
                    self.wake.set()
            if sess is not None and sess.done:
                await final_ev.wait()
        finally:
            if sess is not None:
                self.sessions.pop(sess.sid, None)  # frees caches and decoder state

    def _add(self, ws, lang, ev):
        s = self.eng.new_session(lang)
        s.final_ev = ev
        self.sessions[s.sid] = (s, ws)
        return s

    async def scheduler(self):
        loop = asyncio.get_running_loop()
        while True:
            live = [s for s, _ in self.sessions.values()]
            groups = self.eng.ready_groups(live)
            if not groups:
                self.wake.clear()
                await self.wake.wait()
                continue
            for kind, g in groups:
                t0 = time.perf_counter()
                res = await loop.run_in_executor(self.pool, self.eng.run_group, kind, g)
                dt = time.perf_counter() - t0
                if self.step_log:
                    self.step_log.write(json.dumps({"t": time.time(), "kind": kind, "B": len(g),
                                                    "ms": round(dt * 1000, 3), "live": len(live)}) + "\n")
                for s, text, final, n in res:
                    entry = self.sessions.get(s.sid)
                    if entry is None:
                        continue
                    ws = entry[1]
                    try:
                        await ws.send(json.dumps({"text": text, "n": n, "final": final}, ensure_ascii=False))
                    except Exception:
                        pass
                    if final:
                        s.final_ev.set()
            if self.step_log:
                self.step_log.flush()


async def serve_main(a):
    from websockets.asyncio.server import serve

    eng = BatchedStreamingASR(a.model, a.context_right, dtype=a.dtype, max_batch=a.max_batch,
                              allow_untrained=a.allow_untrained)
    srv = Server(eng, a.step_log)
    # warm-up: one dummy stream through first/next/final steps
    s = eng.new_session("en")
    s.accept(np.zeros(16000, np.float32))
    s.input_done()
    while (g := eng.ready_groups([s])):
        for k, grp in g:
            eng.run_group(k, grp)
    asyncio.create_task(srv.scheduler())
    async with serve(srv.handler, "127.0.0.1", a.port, max_size=None, ping_interval=None, max_queue=None):
        print(f"READY chunk={eng.chunk} chunk_ms={eng.chunk_ms} pre_cache={eng.pre_cache}", flush=True)
        await asyncio.Future()


# ------------------------------------------------------------------ self-test against NeMo's own loop

def selftest(a):
    """Decode K utterances (a) through BatchedStreamingASR with all K streams batched and staggered, and (b)
    with NeMo's CacheAwareStreamingAudioBuffer + conformer_stream_step on each utterance alone. Print both."""
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.util import error_rate, load_audio, read_utts

    eng = BatchedStreamingASR(a.model, a.context_right, dtype=a.dtype, allow_untrained=a.allow_untrained)
    utts = read_utts(Path(__file__).resolve().parents[1] / "data" / "test_utts.jsonl")
    utts = utts[: a.selftest // 2] + [u for u in utts if u["lang"] == "zh"][: a.selftest - a.selftest // 2]
    audios = [load_audio(u["path"]) for u in utts]
    # (a) batched, stream i starts i chunks late so batches mix streams at different steps
    sess = [eng.new_session(u["lang"]) for u in utts]
    step = int(eng.chunk_ms * 16)
    pos = [-i * step for i in range(len(utts))]
    while not all(s.finished for s in sess):
        for i, s in enumerate(sess):
            if s.done:
                continue
            if pos[i] >= 0:
                s.accept(audios[i][pos[i] : pos[i] + step])
                if pos[i] + step >= len(audios[i]):
                    s.input_done()
            pos[i] += step
        for k, g in eng.ready_groups(sess):
            eng.run_group(k, g)
    batched = [s.text for s in sess]
    # (b) NeMo reference, one utterance at a time
    m = eng.model
    ref_txt = []
    for u, x, s in zip(utts, audios, sess):
        m._row_prompt_idx = torch.tensor([s.prompt_idx])
        buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False, pad_and_drop_preencoded=False)
        buf.append_audio(x, stream_id=-1)
        c0, c1, c2 = m.encoder.get_initial_cache_state(batch_size=1)
        hyp = None
        with torch.inference_mode():
            for step_num, (chunk, lens) in enumerate(iter(buf)):
                out = m.conformer_stream_step(
                    processed_signal=chunk.to(eng.device), processed_signal_length=lens.to(eng.device),
                    cache_last_channel=c0, cache_last_time=c1, cache_last_channel_len=c2,
                    keep_all_outputs=buf.is_buffer_empty(), previous_hypotheses=hyp,
                    drop_extra_pre_encoded=0 if step_num == 0 else eng.drop_extra, return_transcription=True)
                _, _, c0, c1, c2, hyp = out[:6]
        ref_txt.append(clean_text(hyp[0].text))
    same = 0
    for u, b, r in zip(utts, batched, ref_txt):
        same += b == r
        print(json.dumps({"id": u["id"], "same": b == r, "batched": b, "nemo_loop": r, "ref": u["text"]},
                         ensure_ascii=False))
    print(json.dumps({"identical": same, "of": len(utts),
                      "batched_err": {l: error_rate((u["text"], b, l) for u, b in zip(utts, batched) if u["lang"] == l) for l in ("en", "zh")},
                      "nemo_loop_err": {l: error_rate((u["text"], r, l) for u, r in zip(utts, ref_txt) if u["lang"] == l) for l in ("en", "zh")}},
                     ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=6010)
    ap.add_argument("--context-right", type=int, default=3,
                    help="att_context_size [56, r]; chunk = (r+1)*80 ms; trained: r in {0, 3, 6, 13}")
    ap.add_argument("--allow-untrained", action="store_true")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--step-log", default=None, help="append one JSON line per batched step")
    ap.add_argument("--selftest", type=int, default=0)
    a = ap.parse_args()
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
    if a.selftest:
        selftest(a)
    else:
        asyncio.run(serve_main(a))


if __name__ == "__main__":
    main()
