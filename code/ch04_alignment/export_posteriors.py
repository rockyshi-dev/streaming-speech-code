"""Chapter 4: save one real CTC posterior matrix, so CTC greedy decoding can be done by hand.

    CUDA_VISIBLE_DEVICES= python code/ch04_alignment/export_posteriors.py     (ssh-nemo env, CPU)

Runs the CTC head of NVIDIA's cache-aware FastConformer hybrid model (att_context_size [70,13]) on
one LibriSpeech utterance and writes $BOOK_DATA/runs/ch04/posteriors.npz:
log_probs (T x V, one row per 80 ms encoder frame), the token vocabulary (blank last), and NeMo's own
CTC greedy transcript for comparison. ctc_greedy_by_hand.py reads it in the CPU env.
"""
import os

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this repository's code/, ahead of any installed hbasr
from hbasr import audio, paths  # noqa: E402
from hbasr.backends_nemo import load_model  # noqa: E402

OUT = f"{paths.BOOK_DATA}/runs/ch04"
WAV = f"{paths.BOOK_DATASETS}/librispeech/extracted/LibriSpeech/test-clean/1089/134686/1089-134686-0006.flac"
os.makedirs(OUT, exist_ok=True)
m = load_model("fastconformer-en-multi", "cpu")
m.encoder.set_default_att_context_size([70, 13])
m.change_decoding_strategy(decoder_type="ctc", verbose=False)
x, sr = audio.read_wav(WAV)
with torch.inference_mode():
    sig = torch.from_numpy(x)[None]
    enc, enc_len = m.forward(input_signal=sig, input_signal_length=torch.tensor([x.size]))[:2]
    lp = m.ctc_decoder(encoder_output=enc)  # (1, T, V+1) log-probabilities, blank = last index
    hyp = m.ctc_decoding.ctc_decoder_predictions_tensor(lp, decoder_lengths=enc_len, return_hypotheses=False)
h = hyp[0][0] if isinstance(hyp, tuple) else hyp[0]
text = h.text if hasattr(h, "text") else str(h)
vocab = list(m.tokenizer.vocab) if hasattr(m.tokenizer, "vocab") else [m.tokenizer.ids_to_tokens([i])[0]
                                                                        for i in range(m.tokenizer.vocab_size)]
np.savez_compressed(os.path.join(OUT, "posteriors.npz"), log_probs=lp[0, : int(enc_len[0])].numpy(),
                    vocab=np.array(vocab + ["<blank>"]), nemo_text=text, wav=WAV, frame_ms=80)
print(lp.shape, int(enc_len[0]), text)
