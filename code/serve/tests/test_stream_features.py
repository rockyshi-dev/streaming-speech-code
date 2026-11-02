import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.stream_features import StreamFeaturizer, reference_mel  # noqa: E402


@pytest.mark.parametrize("chunk", [160, 320, 2560, 997, 7])
def test_streaming_equals_offline(chunk):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000 * 2 + 123).astype(np.float32) * 0.1
    offline = reference_mel(x)
    f = StreamFeaturizer(reference_mel, keep_s=0.5)
    parts = [f.accept(x[i : i + chunk]) for i in range(0, len(x), chunk)]
    parts.append(f.finish())
    stream = np.concatenate([p for p in parts if p.size], axis=0)
    assert stream.shape == offline.shape
    np.testing.assert_allclose(stream, offline, rtol=1e-5, atol=1e-4)


def test_frames_only_when_window_complete():
    f = StreamFeaturizer(reference_mel)
    assert f.accept(np.zeros(255, np.float32)).size == 0  # frame 0 needs 256 samples
    assert f.accept(np.zeros(1, np.float32)).shape[0] == 1
    assert f.accept(np.zeros(159, np.float32)).size == 0
    assert f.accept(np.zeros(1, np.float32)).shape[0] == 1


def test_audio_is_trimmed():
    f = StreamFeaturizer(reference_mel)
    for _ in range(100):
        f.accept(np.zeros(2560, np.float32))
    assert len(f.audio) < 2560 + 3 * 160 + 256


def test_count_last_frame_false_drops_only_the_final_frame():
    x = np.random.default_rng(1).standard_normal(16000).astype(np.float32)
    f = StreamFeaturizer(reference_mel, count_last_frame=False)
    parts = [f.accept(x[i : i + 2560]) for i in range(0, len(x), 2560)] + [f.finish()]
    stream = np.concatenate([p for p in parts if p.size])
    off = reference_mel(x)
    assert stream.shape[0] == off.shape[0] - 1
    np.testing.assert_allclose(stream, off[:-1], rtol=1e-5, atol=1e-4)


def test_against_nemo_front_end():
    """The real NeMo front end (CPU) through StreamFeaturizer equals NeMo's offline features. Skipped where
    NeMo is not installed (run it with the ssh-serve-nemo environment)."""
    nemo_feat = pytest.importorskip("nemo.collections.asr.modules")
    import torch

    pre = nemo_feat.AudioToMelSpectrogramPreprocessor(sample_rate=16000, normalize="NA", window_size=0.025,
                                                      window_stride=0.01, window="hann", features=128,
                                                      n_fft=512, log=True, dither=0.0, pad_to=0)
    pre.eval()

    def mel(seg):
        with torch.inference_mode():
            f, _ = pre(input_signal=torch.from_numpy(np.ascontiguousarray(seg))[None], length=torch.tensor([len(seg)]))
        return f[0].T.numpy()

    x = (np.random.default_rng(2).standard_normal(16000 * 3 + 77) * 0.05).astype(np.float32)
    with torch.inference_mode():
        off, n = pre(input_signal=torch.from_numpy(x)[None], length=torch.tensor([len(x)]))
    off = off[0, :, : int(n[0])].T.numpy()
    f = StreamFeaturizer(mel, count_last_frame=False)
    parts = [f.accept(x[i : i + 2560]) for i in range(0, len(x), 2560)] + [f.finish()]
    stream = np.concatenate([p for p in parts if p.size])
    assert stream.shape == off.shape
    np.testing.assert_allclose(stream, off, rtol=1e-4, atol=2e-4)
