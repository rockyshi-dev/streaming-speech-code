from hbasr import textnorm
from hbasr.backends_sherpa import char_times, word_start_times


def test_word_start_times():
    toks = [" AFTER", " E", "AR", "LY", " NIGHT"]
    assert word_start_times(toks, [0.1, 0.2, 0.3, 0.4, 0.5]) == [0.1, 0.2, 0.5]


def test_char_times_with_byte_fallback():
    toks = [" 其", "中", " ", "深", "<0xE5>", "<0x9C>", "<0xB3>", " 甚"]
    t = [0.1, 0.2, 0.25, 0.3, 0.4, 0.42, 0.44, 0.5]
    out = char_times(toks, t, lambda s: textnorm.zh_units(textnorm.ZH_DEFAULT(s)))
    assert out == [0.1, 0.2, 0.3, 0.4, 0.5]   # 其 中 深 圳 甚
