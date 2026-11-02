import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.handoff import SentenceAggregator  # noqa: E402


def stream(agg, text, step=3):
    out = []
    for i in range(0, len(text), step):
        out += agg.feed(text[i : i + step])
    return out + agg.flush()


def test_english_sentences():
    segs = stream(SentenceAggregator(), "Sure. The museum opens at 9.30 today! Want directions?")
    assert segs == ["Sure.", "The museum opens at 9.30 today!", "Want directions?"]


def test_decimal_point_is_not_a_sentence_end():
    agg = SentenceAggregator()
    assert agg.feed("It costs 3.") == []
    assert agg.feed("50 dollars. Ok") == ["It costs 3.50 dollars."]
    assert agg.flush() == ["Ok"]


def test_mandarin_sentences():
    segs = stream(SentenceAggregator(), "好的。今天天气很好！你想去哪里？")
    assert segs == ["好的。", "今天天气很好！", "你想去哪里？"]


def test_first_segment_comma_cut():
    agg = SentenceAggregator(first_min_chars=10)
    segs = stream(agg, "Well, that is a good question, and here is the answer. Second one, not cut.")
    assert segs[0] == "Well, that is a good question,"
    assert segs[-1] == "Second one, not cut."


def test_markdown_is_dropped():
    assert stream(SentenceAggregator(), "**Yes.** #1 choice.") == ["Yes.", "1 choice."]
