import math

from mp4translator.text import count_hu_syllables
from mp4translator.tts import plausible_duration

TEXT = "Szia, ez egy teszt mondat."


def expected(text=TEXT, sps=6.0):
    return max(count_hu_syllables(text) / sps, 0.4)


def test_zero_at_expected():
    assert plausible_duration(expected(), TEXT, 6.0) < 1e-9


def test_symmetric_in_log_space():
    e = expected()
    assert math.isclose(plausible_duration(e * 2, TEXT, 6.0), plausible_duration(e / 2, TEXT, 6.0))
    assert math.isclose(plausible_duration(e * 2, TEXT, 6.0), math.log(2))


def test_nonpositive_duration_is_inf():
    assert plausible_duration(0, TEXT, 6.0) == math.inf
    assert plausible_duration(-1.0, TEXT, 6.0) == math.inf


def test_minimum_expected_is_point_four():
    assert plausible_duration(0.4, "", 6.0) < 1e-9
    assert plausible_duration(0.4, "x", 6.0) < 1e-9


def _fake_xtts(sps_of_voice):
    import numpy as np

    from mp4translator.config import Config
    from mp4translator.tts import XTTSEngine

    engine = object.__new__(XTTSEngine)  # modell betöltése nélkül
    engine.cfg = Config(syllables_per_second=6.0, xtts_retries=3)
    engine.sample_rate = 1000
    engine.latents = {"SPK0": None}
    engine.rates = {}
    engine.last_attempts = 0
    engine._generate = lambda text, speaker, speed: np.zeros(
        int(count_hu_syllables(text) / sps_of_voice * 1000), dtype=np.float32
    )
    return engine


LONG_TEXT = "Ma reggel korán keltem, megittam egy kávét, aztán elindultam."


def test_slow_voice_retries_without_measured_rate(tmp_path):
    engine = _fake_xtts(3.6)  # a beállított 6,0-hoz képest 1,67× hosszabb → gyanús
    engine.synthesize(LONG_TEXT, "SPK0", tmp_path / "a.wav")
    assert engine.last_attempts == 3


def test_measured_rate_avoids_needless_retries(tmp_path):
    engine = _fake_xtts(3.6)
    engine.set_rate("SPK0", 4.0)
    engine.synthesize(LONG_TEXT, "SPK0", tmp_path / "a.wav")
    assert engine.last_attempts == 1
    engine.set_rate("SPK0", None)
    assert engine.rates == {}


def test_silent_attempts_do_not_crash(tmp_path):
    import numpy as np

    engine = _fake_xtts(6.0)
    engine._generate = lambda text, speaker, speed: np.zeros(0, dtype=np.float32)  # mindig üres hang
    assert engine.synthesize(LONG_TEXT, "SPK0", tmp_path / "a.wav") == 0.0
    assert engine.last_attempts == 3
    assert (tmp_path / "a.wav").exists() and not (tmp_path / "a.part.wav").exists()
