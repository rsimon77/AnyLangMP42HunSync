from pathlib import Path

from mp4translator.config import Config
from mp4translator.manifest import Manifest, Speaker


def test_defaults():
    c = Config()
    assert c.background == "separate"
    assert c.tts == "xtts"
    assert c.xtts_accept_license is False


def test_load_repo_config():
    path = Path(__file__).resolve().parent.parent / "config.yaml"
    assert path.exists()
    assert isinstance(Config.load(path), Config)


def test_load_missing_returns_defaults(tmp_path):
    assert Config.load(tmp_path / "nope.yaml") == Config()


def test_speakers_roundtrip(tmp_path):
    m = Manifest(
        source="in.mp4",
        speakers={"SPK0": Speaker(ref_audio="speakers/SPK0.wav", seconds=12.5), "SPK1": Speaker()},
    )
    p = tmp_path / "manifest.json"
    m.save(p)
    m2 = Manifest.load(p)
    assert m2.speakers == m.speakers
    assert m2.speakers["SPK0"].seconds == 12.5
    assert m2.speakers["SPK1"].ref_audio is None
