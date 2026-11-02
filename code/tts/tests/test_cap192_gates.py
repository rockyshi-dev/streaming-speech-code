"""The cap-192 gate checker's helpers: tolerances, the anomaly share of G6d, and the rule that missing evidence is not a pass."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch14_harness"))
import check_cap192_gates as g  # noqa: E402


def test_within_is_false_for_missing_values():
    assert g.within(1.1, 1.0, 0.15) and not g.within(1.2, 1.0, 0.15)
    assert not g.within(None, 1.0, 0.1) and not g.within(1.0, None, 0.1)


def test_anomaly_share_uses_the_baseline_medians():
    base = [{"lang": "en", "id": "a", "audio_s": s} for s in (5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 10.0)]
    med, thr = g.baseline_item_stats(base)
    assert med[("en", "a")] == 5.0 and thr == 1.0  # the one 10 s request deviates by 1.0 from the median
    rows = [{"lang": "en", "id": "a", "audio_s": s, "visit": 3} for s in (5.0, 5.0, 2.0, 5.0)]
    assert g.anomaly_share(rows, 3, med, 0.5) == 0.25  # only the 2 s request is further than 0.5 from 5 s
    assert g.anomaly_share(rows, 4, med, 0.5) is None  # no rows for that visit: no evidence


def test_log_without_timestamps_in_the_window_gives_none_not_zero():
    assert g.log_hits("no stamps here, preempted", 0, 1, 2026) is None


def test_ecc_gate_needs_two_readings_and_no_change():
    assert g.gate_ecc("DRAM Uncorrectable : 0\nDRAM Uncorrectable : 0")[0]
    assert not g.gate_ecc("DRAM Uncorrectable : 0")[0]
    assert not g.gate_ecc("DRAM Uncorrectable : 0\nDRAM Uncorrectable : 2")[0]
    assert g.gate_ecc("DRAM Uncorrectable : 1\nDRAM Uncorrectable : 1")[0]  # a card that logged an error earlier: no new one
