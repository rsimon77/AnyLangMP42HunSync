import pytest

from mp4translator.config import Config
from mp4translator.steps.align import plan_fit

CFG = Config()


def fit(tts, can_shorten=False, start=0.0, end=2.0, next_start=3.0):
    return plan_fit(tts, start, end, next_start, CFG, can_shorten)


def test_ok():
    assert fit(2.0).action == "ok"
    assert fit(1.0).speed == 1.0


def test_borrow():
    # available = 2 + 1 * 0.8 = 2.8
    assert fit(2.5).action == "borrow"
    assert fit(2.8).action == "borrow"


def test_speedup():
    f = fit(3.0)
    assert f.action == "speedup"
    assert f.speed == pytest.approx(3.0 / 2.8)


def test_speedup_boundary_prefers_speedup_over_shorten():
    f = fit(2.8 * 1.15, can_shorten=True)
    assert f.action == "speedup"


def test_shorten():
    assert fit(3.5, can_shorten=True).action == "shorten"


def test_forced_speedup():
    f = fit(3.5)
    assert f.action == "forced_speedup"
    assert f.speed == pytest.approx(3.5 / 2.8)


def test_overflow():
    f = fit(4.0)
    assert f.action == "overflow"
    assert f.speed == CFG.max_speedup


def test_shorten_takes_precedence_over_overflow():
    assert fit(10.0, can_shorten=True).action == "shorten"


def test_last_segment_no_gap():
    # next_start == video end == segment end -> no borrowing
    assert plan_fit(2.2, 0, 2.0, 2.0, CFG, False).action == "speedup"
    assert plan_fit(2.2, 0, 2.0, 2.0, CFG, False).speed == pytest.approx(1.1)


def test_last_segment_with_trailing_silence_borrows():
    assert plan_fit(2.5, 0, 2.0, 3.0, CFG, False).action == "borrow"


def test_negative_gap_is_ignored():
    f = plan_fit(2.2, 0, 2.0, 1.5, CFG, False)
    assert f.action == "speedup"
    assert f.speed == pytest.approx(1.1)
