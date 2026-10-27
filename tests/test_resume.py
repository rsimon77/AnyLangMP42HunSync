import os

import pytest

from mp4translator import runlock
from mp4translator.manifest import Segment
from mp4translator.steps.synthesize import is_synthesized


def _done_segment(tmp_path, **overrides):
    (tmp_path / "tts").mkdir(exist_ok=True)
    (tmp_path / "tts" / "seg_0000.wav").write_bytes(b"x")
    data = dict(
        id=0, start=0.0, end=1.0, speaker="SPK0", text_en="Hi", text_hu="Szia",
        tts_file="tts/seg_0000.wav", tts_text="Szia", tts_speaker="SPK0", action="ok",
    )
    data.update(overrides)
    return Segment(**data)


def test_is_synthesized_when_everything_matches(tmp_path):
    assert is_synthesized(_done_segment(tmp_path), tmp_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"text_hu": "Helló"},  # kézzel átírt fordítás
        {"speaker": "SPK1"},  # más beszélőhöz rendelve
        {"tts_text": None},  # a voices lépés érvénytelenítette
        {"action": None},  # félbemaradt szegmens
        {"tts_file": "tts/missing.wav"},
    ],
)
def test_is_synthesized_detects_stale_audio(tmp_path, overrides):
    assert not is_synthesized(_done_segment(tmp_path, **overrides), tmp_path)


def test_lock_acquire_and_release(tmp_path):
    runlock.acquire(tmp_path)
    assert runlock.running_pid(tmp_path) == os.getpid()
    runlock.release(tmp_path)
    assert runlock.running_pid(tmp_path) is None


def test_lock_refuses_second_live_process(tmp_path, monkeypatch):
    monkeypatch.setattr(runlock, "_is_ours", lambda pid: True)
    (tmp_path / runlock.LOCK_NAME).write_text(str(os.getppid()))  # élő, de másik mp4translator-folyamat
    with pytest.raises(runlock.AlreadyRunning):
        runlock.acquire(tmp_path)


def test_lock_with_reused_pid_is_stale(tmp_path):
    # Az elárvult zárfájl PID-jét a rendszer egy másik programnak adta: nem futás, nem is szabad leállítani
    (tmp_path / runlock.LOCK_NAME).write_text(str(os.getppid()))
    assert runlock.running_pid(tmp_path) is None
    assert runlock.request_stop(tmp_path) is None
    runlock.acquire(tmp_path)
    assert runlock.running_pid(tmp_path) == os.getpid()


def test_stale_lock_is_cleaned_up(tmp_path):
    (tmp_path / runlock.LOCK_NAME).write_text("999999")
    assert runlock.running_pid(tmp_path) is None
    assert not (tmp_path / runlock.LOCK_NAME).exists()
    runlock.acquire(tmp_path)
    runlock.release(tmp_path)


def test_request_stop_without_run(tmp_path):
    assert runlock.request_stop(tmp_path) is None
