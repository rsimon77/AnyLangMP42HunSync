import numpy as np
import pytest
import soundfile as sf

from mp4translator.config import Config
from mp4translator.manifest import KeptSound, Manifest, Segment, Speaker
from mp4translator.steps import sounds
from mp4translator.steps.translate import speakers_note
from mp4translator.tts import EdgeTTSEngine
from mp4translator.voice_kind import classify_kind, median_f0, pitch_correction


def tone(f0, seconds=1.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    # harmonikusokkal, hogy a pyin biztosan zöngésnek lássa
    return (0.3 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(4 * np.pi * f0 * t)).astype(np.float32)


@pytest.mark.parametrize("f0", [120, 220, 400])
def test_median_f0_finds_pitch(f0):
    assert median_f0(tone(f0), 16000) == pytest.approx(f0, rel=0.03)


def test_median_f0_of_silence_is_none():
    assert median_f0(np.zeros(16000, dtype=np.float32), 16000) is None


@pytest.mark.parametrize(
    "f0,kind",
    [(110, "male"), (164, "male"), (165, "female"), (284, "female"), (300, "child"), (464, "child"), (None, None)],
)
def test_classify_kind(f0, kind):
    assert classify_kind(f0, Config()) == kind


def test_pitch_correction_only_for_children_by_default():
    cfg = Config()
    assert pitch_correction(408, 282, "child", cfg) == pytest.approx(6.39, abs=0.01)
    assert pitch_correction(297, 236, "female", cfg) == 0.0


def test_pitch_correction_all_threshold_and_cap():
    cfg = Config(pitch_correction="all")
    assert pitch_correction(297, 236, "female", cfg) == pytest.approx(3.98, abs=0.01)
    assert pitch_correction(284, 270, "female", cfg) == 0.0  # 0,9 félhang: nem korrigáljuk
    assert pitch_correction(408, 253, "child", cfg) == cfg.max_pitch_shift  # 8,3 félhang → határ
    assert pitch_correction(150, 200, "male", cfg) == pytest.approx(-4.98, abs=0.01)  # lefelé is
    assert pitch_correction(None, 200, "male", cfg) == 0.0
    assert pitch_correction(408, 282, "child", Config(pitch_correction="off")) == 0.0


def test_edge_picks_voice_by_kind():
    tts = EdgeTTSEngine("hu-HU-TamasNeural", 44100, Config().edge_voices, "+25Hz")
    tts.prepare_speaker("A", None, "female")
    tts.prepare_speaker("B", None, "child")
    tts.prepare_speaker("C", None, None)
    assert tts.speakers == {
        "A": ("hu-HU-NoemiNeural", "+0Hz"),
        "B": ("hu-HU-NoemiNeural", "+25Hz"),
        "C": ("hu-HU-TamasNeural", "+0Hz"),
    }


def test_speakers_note_for_prompt():
    m = Manifest(source="a.mp4", speakers={"SPK0": Speaker(kind="female"), "SPK1": Speaker(kind="child"), "SPK2": Speaker()})
    assert speakers_note(m) == 'Speakers: {"SPK0": "adult woman", "SPK1": "child"}\n\n'
    assert speakers_note(Manifest(source="a.mp4")) == ""


def test_speech_gaps_leave_margin_and_skip_short():
    gaps = sounds.speech_gaps([(1.0, 2.0), (2.4, 4.0), (6.0, 7.0)], duration=10.0)
    # 0–0,85 · (2,0–2,4 túl rövid) · 4,15–5,85 · 7,15–10
    assert gaps == [(0.0, 0.85), (4.15, 5.85), (7.15, 10.0)]


def test_windows_cover_gap():
    assert sounds.windows([(0.0, 3.5)]) == [(0.0, 2.0), (1.0, 3.0), (2.0, 3.5)]
    assert sounds.windows([(5.0, 5.6)]) == [(5.0, 5.6)]


def test_merge_hits_keeps_strongest_label():
    merged = sounds.merge_hits([(0, 2, "Bark", 0.4), (1, 3, "Dog", 0.7), (10, 12, "Meow", 0.5)])
    assert [(k.start, k.end, k.label) for k in merged] == [(0, 3, "Dog"), (10, 12, "Meow")]


def test_mix_adds_kept_sounds_from_vocals(tmp_path):
    from rich.console import Console

    from mp4translator.context import Context
    from mp4translator.steps.mix import kept_sounds

    sr = 44100
    sf.write(str(tmp_path / "vocals.wav"), np.full(3 * sr, 0.5, dtype=np.float32), sr)
    m = Manifest(source="a.mp4", kept_sounds=[KeptSound(start=1.0, end=2.0, label="Dog", score=0.9)])
    ctx = Context(Config(), tmp_path / "in.mp4", tmp_path / "out.mp4", tmp_path, m, console=Console(quiet=True))
    out = kept_sounds(ctx, 3 * sr)
    assert out[: sr].max() == 0 and out[2 * sr + 10 :].max() == 0
    assert out[int(1.5 * sr)] == pytest.approx(0.5)


def test_pitch_shift_keeps_duration(tmp_path):
    from mp4translator.audio import duration_of, time_stretch

    sr = 44100
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(200, 2.0, sr), sr)
    time_stretch(src, tmp_path / "b.wav", 1.0, 4.0)
    time_stretch(src, tmp_path / "c.wav", 1.25, 4.0)
    assert duration_of(tmp_path / "b.wav") == pytest.approx(2.0, abs=0.03)
    assert duration_of(tmp_path / "c.wav") == pytest.approx(1.6, abs=0.03)
    y, file_sr = sf.read(str(tmp_path / "b.wav"), dtype="float32")
    assert file_sr == sr
    assert median_f0(y, sr) == pytest.approx(200 * 2 ** (4 / 12), rel=0.03)
