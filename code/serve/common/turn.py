"""Voice activity and end-of-turn decisions for the cascade (chapter 18).

SileroVAD   Silero VAD v6 (MIT), the ONNX file shipped in the silero-vad 6.2.3 wheel, run with onnxruntime on
            32 ms windows (512 samples at 16 kHz) with Silero's 64-sample context, exactly as the package's
            OnnxWrapper does, but without torch.
SmartTurn   Smart Turn v3.2 CPU (BSD-2-Clause, pipecat-ai/smart-turn-v3), with the reference preprocessing of
            pipecat-ai/smart-turn inference.py: last 8 s of audio, zero-padded at the front, Whisper log-mel
            features (80 x 800); the model returns P(turn complete).
Endpointer  The decision logic, independent of the models (unit-tested with made-up probabilities):
            * a window is speech if p >= threshold, silence if p < threshold - 0.15 (Silero's hysteresis);
            * speech starts after `start_ms` of consecutive speech windows;
            * silence is timed like Silero's VADIterator: from the first silence window after speech, with
              windows in the hysteresis band counting toward it, until a speech window resets it;
            * mode "silence": the turn ends once that silence has lasted `silence_ms`;
            * mode "smart": after `trigger_ms` of silence the turn model is asked once; if it says complete
              (p >= 0.5) the turn ends at once, otherwise only speech resuming or `fallback_ms` of silence
              ends the wait (Pipecat's arrangement: a short VAD stop, then Smart Turn, then a long timeout).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import paths

SILERO_ONNX = Path(f"{paths.BOOK_DATA}/vendor/silero/silero_vad/data/silero_vad.onnx")
SMART_TURN_ONNX = Path(f"{paths.BOOK_MODELS}/smart-turn-v3/smart-turn-v3.2-cpu.onnx")
WIN = 512  # 32 ms at 16 kHz
WIN_MS = 32.0


class SileroVAD:
    def __init__(self, path: Path = SILERO_ONNX):
        import onnxruntime as ort

        o = ort.SessionOptions()
        o.inter_op_num_threads = 1
        o.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=o)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, 64), np.float32)

    def __call__(self, x: np.ndarray) -> float:
        assert len(x) == WIN
        inp = np.concatenate([self.context, x.reshape(1, -1).astype(np.float32)], axis=1)
        out, self.state = self.sess.run(None, {"input": inp, "state": self.state, "sr": np.array(16000, np.int64)})
        self.context = inp[:, -64:]
        return float(out[0, 0])

    def probs(self, x: np.ndarray) -> np.ndarray:
        """Offline: probabilities for consecutive windows of x (fresh state)."""
        self.reset()
        return np.array([self(x[i : i + WIN]) for i in range(0, len(x) - WIN + 1, WIN)])


def speech_end_reference(x: np.ndarray, vad: SileroVAD, threshold: float = 0.5) -> tuple[float, float] | None:
    """Offline reference for the turn's true speech start and end (s): first and last 32 ms window with
    p >= threshold, using the whole file (so it can look at audio a causal endpointer has not seen yet)."""
    p = vad.probs(x)
    idx = np.nonzero(p >= threshold)[0]
    if len(idx) == 0:
        return None
    return float(idx[0] * WIN / 16000), float((idx[-1] + 1) * WIN / 16000)


class SmartTurn:
    def __init__(self, path: Path = SMART_TURN_ONNX):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        o = ort.SessionOptions()
        o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        o.inter_op_num_threads = 1
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])
        self.fe = WhisperFeatureExtractor(chunk_length=8)

    def __call__(self, audio: np.ndarray) -> float:
        n = 8 * 16000
        a = audio[-n:] if len(audio) > n else np.pad(audio, (n - len(audio), 0))
        f = self.fe(a, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=n,
                    truncation=True, do_normalize=True).input_features.astype(np.float32)
        return float(self.sess.run(None, {"input_features": f})[0][0].item())


@dataclass
class Endpointer:
    mode: str = "silence"            # "silence" or "smart"
    threshold: float = 0.5
    start_ms: float = 96.0
    silence_ms: float = 500.0
    trigger_ms: float = 200.0
    fallback_ms: float = 3000.0
    turn_model: Callable[[np.ndarray], float] | None = None
    in_speech: bool = False
    speech_run: int = 0
    silence_run: int = 0
    asked: bool = False
    ended: bool = False
    events: list = field(default_factory=list)

    def reset(self):
        self.in_speech = False
        self.speech_run = self.silence_run = 0
        self.asked = self.ended = False

    def step(self, p: float, t: float, audio_so_far: np.ndarray | None = None) -> list[dict]:
        """Feed the probability of one 32 ms window that ended at time t. Returns new events."""
        ev = []
        speech = p >= self.threshold
        silence = p < self.threshold - 0.15
        if speech:
            self.speech_run += 1
            self.silence_run = 0
            self.asked = False
            if not self.in_speech and self.speech_run * WIN_MS >= self.start_ms:
                self.in_speech = True
                ev.append({"type": "speech_start", "t": t})
        else:
            self.speech_run = 0
            if silence or self.silence_run > 0:  # band windows extend a silence that has started
                self.silence_run += 1
        if self.in_speech and not self.ended and self.silence_run > 0 and not speech:
            sil = self.silence_run * WIN_MS
            if self.mode == "silence":
                if sil >= self.silence_ms:
                    self._end(ev, t, "silence")
            else:
                if sil >= self.trigger_ms and not self.asked:
                    self.asked = True
                    t0 = time.perf_counter()
                    q = self.turn_model(audio_so_far) if self.turn_model else 1.0
                    dt = time.perf_counter() - t0
                    ev.append({"type": "turn_check", "t": t, "p_complete": q, "infer_ms": dt * 1000})
                    if q >= 0.5:
                        self._end(ev, t, "smart_turn")
                if not self.ended and sil >= self.fallback_ms:
                    self._end(ev, t, "fallback")
        self.events += ev
        return ev

    def _end(self, ev, t, reason):
        self.ended = True
        self.in_speech = False
        ev.append({"type": "endpoint", "t": t, "reason": reason})
