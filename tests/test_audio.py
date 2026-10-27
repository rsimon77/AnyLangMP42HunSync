import numpy as np
import pytest

from mp4translator.audio import duck_envelope, trim_silence

SR = 1000


def test_trim_silence_empty():
    assert trim_silence(np.zeros(0, dtype=np.float32), SR).size == 0


def test_trim_silence_all_silent():
    assert trim_silence(np.zeros(500, dtype=np.float32), SR).size == 0


def test_trim_silence_removes_edges_with_padding():
    y = np.zeros(1000, dtype=np.float32)
    y[400:600] = 0.5
    out = trim_silence(y, SR, pad_s=0.03)  # pad = 30 samples
    assert out.size == 200 + 2 * 30
    assert out[30:-30].min() == 0.5


def test_trim_silence_padding_clamped_to_bounds():
    y = np.full(100, 0.5, dtype=np.float32)
    assert trim_silence(y, SR, pad_s=1.0).size == 100


def test_trim_silence_ignores_quiet_noise_relative_to_peak():
    y = np.zeros(1000, dtype=np.float32)
    y[100:200] = 1e-4  # -80 dB vs peak
    y[500:600] = 1.0
    out = trim_silence(y, SR, pad_s=0.0)
    assert out.size == 100


def test_duck_envelope_length_and_dtype():
    env = duck_envelope([(1.0, 2.0)], 5 * SR, SR, -20.0)
    assert env.shape == (5 * SR,)
    assert env.dtype == np.float32


def test_duck_envelope_no_intervals_is_unity():
    env = duck_envelope([], 2 * SR, SR, -20.0)
    assert np.allclose(env, 1.0)


def test_duck_envelope_levels():
    duck_db = -20.0
    gain = 10 ** (duck_db / 20)
    env = duck_envelope([(4.0, 6.0)], 10 * SR, SR, duck_db)
    assert env[1 * SR] == pytest.approx(1.0, abs=1e-4)
    assert env[9 * SR] == pytest.approx(1.0, abs=1e-4)
    assert env[5 * SR] == pytest.approx(gain, abs=1e-4)


def test_duck_envelope_bounded():
    duck_db = -22.0
    gain = 10 ** (duck_db / 20)
    env = duck_envelope([(0.0, 1.0), (3.0, 4.5), (9.5, 12.0)], 10 * SR, SR, duck_db)
    assert env.min() >= gain - 1e-5
    assert env.max() <= 1.0 + 1e-5
