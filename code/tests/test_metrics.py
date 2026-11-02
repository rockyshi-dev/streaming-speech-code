import jiwer
import numpy as np

from hbasr import metrics, textnorm


def test_edit_counts_vs_jiwer():
    rng = np.random.default_rng(0)
    vocab = list("abcdef")
    for _ in range(200):
        r = [vocab[i] for i in rng.integers(0, 6, rng.integers(1, 9))]
        h = [vocab[i] for i in rng.integers(0, 6, rng.integers(0, 9))]
        ec = metrics.edit_counts(r, h)
        if h:
            o = jiwer.process_words(" ".join(r), " ".join(h))
            assert ec.errors == o.substitutions + o.deletions + o.insertions
        else:
            assert ec.errors == len(r)


def test_corpus_rate_sums_counts():
    pairs = [("a b c d", "a b c d"), ("a b", "a x")]
    ec = metrics.corpus_rate(pairs, str.split, lambda s: s)
    assert ec.ref_len == 6 and ec.errors == 1


def test_stability():
    hyps = [["the"], ["the", "cat"], ["a", "cat", "sat"], ["a", "cat", "sat"]]
    st = metrics.stability(hyps)
    assert st.unstable_units == 2 and st.revisions == 1 and st.final_units == 3
    assert metrics.stability([["x"], ["x", "y"]]).unstable_units == 0  # appending is stable
    agg = metrics.upwr_upsr([st, metrics.stability([["q"]])])
    assert abs(agg["UPWR"] - 2 / 4) < 1e-12 and agg["UPSR"] == 0.5


def test_aligned_ref_errors():
    assert metrics.aligned_ref_errors("a b c d".split(), "a x c".split()) == [1, 3]


def test_zh_normalizer():
    n = textnorm.ZH_DEFAULT
    assert n("其中，广州 深圳！") == "其中广州深圳"
    assert n("ＡＢＣ") == "abc"
    assert n("廣州") == "广州"
    assert n("2019年") == "二零一九年"
    assert n("即便是。 <zh-CN>") == "即便是"
    assert textnorm.zh_units("我爱 NLP 和AI") == ["我", "爱", "NLP", "和", "AI"]


def test_en_normalizer():
    n = textnorm.EN_DEFAULT
    assert n("Hello, World! It's") == "hello world it's"
    assert textnorm.EN["raw"]("Hello  World") == "Hello World"


def test_matched_pairs():
    assert metrics.matched_pairs("a b c d".split(), "a x c d e".split()) == [(0, 0), (2, 2), (3, 3)]
