import numpy as np

from hbasr import audio


def test_arithmetic():
    assert audio.samples(5, 16000) == 80000
    assert audio.pcm_bytes(1, 16000) == 32000
    assert audio.pcm_bytes(0.16, 16000) == 5120
    assert audio.pcm_bitrate_kbps(24000) == 384.0
    assert audio.samples(0.02, 48000) == 960


def test_int16_roundtrip_and_layout():
    x = np.array([0.0, 0.5, -0.5, 0.999], np.float32)
    x16 = audio.float_to_int16(x)
    assert x16.tolist() == [0, 16384, -16384, 32734]
    b = audio.to_le_bytes(x16)
    assert b[2:4] == bytes([0x00, 0x40])          # 16384 little-endian
    assert audio.from_bytes(b, ">i2")[1] == 64      # same bytes read big-endian
    assert audio.from_bytes(b, "<f4").size == 2     # float32 read: half as many samples


def test_chunks():
    x = np.zeros(16000 + 100, np.float32)
    cs = list(audio.chunks(x, 16000, 160))
    assert len(cs) == 7 and cs[0].samples.size == 2560 and cs[-1].samples.size == 16100 - 6 * 2560
    cs = list(audio.chunks(x, 16000, 160, pad_last=True))
    assert all(c.samples.size == 2560 for c in cs)


def test_naive_decimation_aliases():
    t = audio.tone(10000, 1.0, 48000) + audio.tone(1000, 1.0, 48000)
    assert audio.band_energy_db(audio.naive_decimate(t, 3), 16000, 5800, 6200) > -4
    assert audio.band_energy_db(audio.resample(t, 48000, 16000), 16000, 5800, 6200) < -30


def test_dbfs():
    assert abs(audio.dbfs(audio.tone(1000, 1, 16000, amp=1.0)) + 3.01) < 0.01
    x = audio.tone(1000, 1, 16000)
    assert abs(audio.dbfs(audio.gain_db(x, -6.02)) - audio.dbfs(x) + 6.02) < 1e-3
