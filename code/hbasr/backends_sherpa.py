"""sherpa-onnx streaming transducer adapter for the latency harness (CPU track, chapters 4-9)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from . import paths

MODEL_STORE = str(paths.BOOK_MODELS)
OUT_ROOT = str(paths.BOOK_DATA)

MODELS = {
    # name: (directory under the model store, file stem suffix, modeling unit for hotwords)
    "zipformer-en": ("sherpa-onnx-streaming-zipformer-en-2023-06-26", "epoch-99-avg-1-chunk-16-left-128", "bpe"),
    "zipformer-zh": ("sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12", "epoch-20-avg-1-chunk-16-left-128",
                     "cjkchar+bpe"),
}

# icefall streaming Zipformer exports flush their last chunk only when followed by some audio;
# sherpa-onnx's own examples append 0.66 s of zeros before input_finished(). This tail padding is
# synthetic (never "arrives"), so it costs compute time but no waiting time in the harness.
# The padding is a trade-off, measured on the book's test subsets (chapter 6): with 0.66 s the Mandarin model
# twice stopped two bytes into a three-byte character (AISHELL-1 BAC009S0770W0241 ends in U+FFFD); with
# 1.5 s it completed them but appended words nobody said to other utterances (insertions 21 -> 27 on
# LibriSpeech test-clean greedy, 0 -> 6 on AISHELL-1 beam search; summaries kept in
# ch09_measure/results/tail_padding_1.5s/). The book keeps sherpa-onnx's own 0.66 s.
TAIL_PADDING_S = 0.66


@dataclass
class SherpaConfig:
    model: str = "zipformer-en"
    int8: bool = False
    decoding_method: str = "greedy_search"   # or "modified_beam_search"
    max_active_paths: int = 4
    num_threads: int = 1
    hotwords_file: str = ""
    hotwords_score: float = 1.5
    enable_endpoint: bool = False
    rule1_min_trailing_silence: float = 2.4
    rule2_min_trailing_silence: float = 1.2
    rule3_min_utterance_length: float = 20.0
    tail_padding_s: float = TAIL_PADDING_S

    def files(self) -> dict:
        d, stem, _ = MODELS[self.model]
        base = os.path.join(MODEL_STORE, d)
        q = ".int8" if self.int8 else ""
        return {
            "dir": base,
            "tokens": os.path.join(base, "tokens.txt"),
            "encoder": os.path.join(base, f"encoder-{stem}{q}.onnx"),
            "decoder": os.path.join(base, f"decoder-{stem}{q}.onnx"),
            "joiner": os.path.join(base, f"joiner-{stem}{q}.onnx"),
            "bpe_vocab": os.path.join(OUT_ROOT, "derived", d, "bpe.vocab"),
            "bpe_model": os.path.join(base, "bpe.model"),
        }

    def to_dict(self) -> dict:
        return asdict(self) | {"files": self.files()}


def ensure_bpe_vocab(bpe_model: str, out_path: str) -> str:
    """sherpa-onnx hotwords need a `bpe.vocab` (piece<TAB>score) next to the model; icefall exports
    ship only bpe.model, so derive it with sentencepiece. Written under $BOOK_DATA/derived/, never into
    $BOOK_MODELS."""
    if os.path.exists(out_path):
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=bpe_model)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")
    return out_path


class SherpaRecognizer:
    """One sherpa-onnx OnlineRecognizer; `start()` opens a fresh stream each utterance."""

    sample_rate = 16000

    def __init__(self, cfg: SherpaConfig, bpe_vocab_path: Optional[str] = None):
        import sherpa_onnx

        self.cfg = cfg
        f = cfg.files()
        kw = {}
        if cfg.hotwords_file:
            kw = dict(hotwords_file=cfg.hotwords_file, hotwords_score=cfg.hotwords_score,
                      modeling_unit=MODELS[cfg.model][2],
                      bpe_vocab=bpe_vocab_path or ensure_bpe_vocab(f["bpe_model"], f["bpe_vocab"]))
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f["tokens"], encoder=f["encoder"], decoder=f["decoder"], joiner=f["joiner"],
            num_threads=cfg.num_threads, sample_rate=16000, feature_dim=80,
            decoding_method=cfg.decoding_method, max_active_paths=cfg.max_active_paths,
            enable_endpoint_detection=cfg.enable_endpoint,
            rule1_min_trailing_silence=cfg.rule1_min_trailing_silence,
            rule2_min_trailing_silence=cfg.rule2_min_trailing_silence,
            rule3_min_utterance_length=cfg.rule3_min_utterance_length,
            **kw,
        )
        self.stream = None
        self._endpoint = False

    def start(self) -> None:
        self.stream = self.rec.create_stream()
        self._endpoint = False

    def _decode(self) -> None:
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)

    def accept(self, x: np.ndarray) -> None:
        self.stream.accept_waveform(self.sample_rate, x)
        self._decode()

    def partial(self) -> str:
        return self.rec.get_result(self.stream).strip()

    def endpoint(self) -> bool:
        if not self._endpoint and self.cfg.enable_endpoint and self.rec.is_endpoint(self.stream):
            self._endpoint = True
        return self._endpoint

    def finish(self) -> str:
        if self.cfg.tail_padding_s > 0:
            self.stream.accept_waveform(self.sample_rate,
                                        np.zeros(int(self.cfg.tail_padding_s * self.sample_rate), np.float32))
        self.stream.input_finished()
        self._decode()
        return self.partial()

    def tokens_and_times(self) -> tuple[list[str], list[float]]:
        r = self.rec.get_result_all(self.stream)
        return list(r.tokens), [float(t) for t in r.timestamps]


def word_start_times(tokens: list[str], times: list[float]) -> list[float]:
    """English BPE: a token starting with a space begins a new word; return each word's first-token time."""
    out = []
    for tok, t in zip(tokens, times):
        if tok.startswith(" ") or not out:
            out.append(t)
    return out


def char_times(tokens: list[str], times: list[float], units) -> list[float]:
    """Mandarin: give every scoring unit inside a token that token's time.

    The multi-zh-hans BPE model falls back to raw UTF-8 bytes for rare characters ('<0xE5>',
    '<0x9C>', '<0xB3>' = one character); consecutive byte tokens are joined and decoded first, and
    the character gets the time of its first byte.
    """
    out = []
    buf, t_buf = b"", None
    for tok, t in zip(tokens, times):
        if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
            if not buf:
                t_buf = t
            buf += bytes([int(tok[3:5], 16)])
            try:
                text = buf.decode("utf-8")
            except UnicodeDecodeError:
                continue
            out += [t_buf] * len(units(text))
            buf = b""
            continue
        out += [t] * len(units(tok))
    return out
