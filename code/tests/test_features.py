import numpy as np
import torch
import torchaudio.compliance.kaldi as K

from hbasr import features


def _sig():
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000
    return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


def test_matches_torchaudio_kaldi():
    x = _sig()
    for snip in (True, False):
        ref = K.fbank(torch.from_numpy(x)[None], num_mel_bins=80, dither=0.0, snip_edges=snip,
                      high_freq=-400.0).numpy()
        mine = features.kaldi_fbank(x, 16000, snip_edges=snip, high_freq=-400.0)
        assert ref.shape == mine.shape
        assert np.abs(ref - mine).max() < 1e-3


def test_frame_counts():
    x = np.zeros(16000, np.float32)
    assert features.frame_signal(x, 400, 160, True).shape[0] == 1 + (16000 - 400) // 160
    assert features.frame_signal(x, 400, 160, False).shape[0] == (16000 + 80) // 160


def test_normalization_leak():
    f = features.kaldi_fbank(_sig(), 16000)
    f[60:] += 5.0  # the "future" is louder
    assert features.prefix_leak(features.utterance_cmvn, f, 50) > 0.1
    assert features.prefix_leak(features.causal_cmvn, f, 50) == 0.0
    assert features.prefix_leak(lambda g: features.sliding_cmn(g, 20), f, 50) == 0.0


def test_causal_cmvn_last_frame_equals_utterance():
    f = features.kaldi_fbank(_sig(), 16000)
    a = features.causal_cmvn(f)[-1]
    b = features.utterance_cmvn(f)[-1]
    assert np.abs(a - b).max() < 1e-3
