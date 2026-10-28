import json
import os

import pytest

from mp4translator import runlock
from mp4translator.cleanup import human, plan, remove
from mp4translator.manifest import Manifest, Segment, Speaker
from mp4translator.webui.jobs import Workspace

NEEDED = ["audio.wav", "asr_16k.wav", "vocals.wav", "background.wav", "dub_vocals.wav", "dub_mix.wav",
          "manifest.json", "report.md", "run.log", "config.yaml", "speakers/SPK0.wav", "speakers/SPK0_cal0.wav",
          "speakers/embeddings.npz", "tts/seg_0000.wav", "tts/seg_0000_fit.wav"]


def _touch(p, data=b"x" * 10, mtime=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    if mtime is not None:
        os.utime(p, (mtime, mtime))


def _job(work, name="clip"):
    wd = work / name
    for f in NEEDED + ["tts/seg_0001.wav"]:
        _touch(wd / f)
    os.utime(wd / "tts/seg_0000_fit.wav", None)  # az illesztett hang frissebb a nyersnél → érvényes
    _touch(wd / "tts/seg_0001_fit.wav", mtime=1)  # régebbi a nyersnél → elavult
    for f in ["dub_mix_raw.wav", "audio_16k.wav", "preview/seg_0000.wav", "tts/seg_0000.part.wav",
              "tts/seg_0099.wav", "speakers/SPK7.wav", "speakers/SPK7_cal0.wav", "manifest.run1.json"]:
        _touch(wd / f)
    segs = [Segment(id=i, start=i, end=i + 1, text_en="x", tts_file=f"tts/seg_{i:04d}.wav") for i in range(2)]
    Manifest(source="a.mp4", segments=segs, speakers={"SPK0": Speaker()}).save(wd / "manifest.json")
    _touch(wd / "manifest.tmp")  # félbemaradt mentés (a save ugyanezt a nevet használja, ezért utána)
    return wd


def test_basic_cleanup_keeps_everything_needed(tmp_path):
    work = tmp_path / "work"
    wd = _job(work)
    items = plan(work)
    got = {(i.path.relative_to(wd).as_posix(), i.reason) for i in items}
    assert got == {
        ("dub_mix_raw.wav", "mix_raw"), ("audio_16k.wav", "audio16k"), ("preview", "preview"),
        ("tts/seg_0000.part.wav", "partial"), ("manifest.tmp", "partial"), ("tts/seg_0099.wav", "orphan_tts"),
        ("tts/seg_0001_fit.wav", "stale_fit"), ("speakers/SPK7.wav", "old_speaker"),
        ("speakers/SPK7_cal0.wav", "old_speaker"),
    }
    assert remove(items) == sum(i.size for i in items) > 0
    for f in NEEDED + ["tts/seg_0001.wav", "manifest.run1.json"]:
        assert (wd / f).exists(), f
    assert plan(work) == []


def test_audio16k_kept_without_separated_speech(tmp_path):
    work = tmp_path / "work"
    wd = _job(work)
    (wd / "asr_16k.wav").unlink()  # háttér-szétválasztás nélkül az átírás ezt használja
    assert not any(i.reason == "audio16k" for i in plan(work))


def test_extra_removes_foreign_folders_and_backups_but_not_jobs(tmp_path):
    work = tmp_path / "work"
    wd = _job(work)
    _touch(work / "clips/a.mp4")
    _touch(work / "old_run.log")
    _touch(work / "ollama.log")
    _touch(work / "uploaded/job.json")  # feltöltött, még el nem indított munka
    reasons = {(i.path.name, i.reason) for i in plan(work, extra=True)}
    assert {("clips", "foreign"), ("old_run.log", "foreign"), ("manifest.run1.json", "backup")} <= reasons
    assert not {"ollama.log", "uploaded", wd.name} & {name for name, _ in reasons}


def test_corrupt_manifest_leaves_voice_files_alone(tmp_path):
    work = tmp_path / "work"
    wd = _job(work)
    (wd / "manifest.json").write_text("{nem json")
    reasons = {i.reason for i in plan(work)}
    assert reasons.isdisjoint({"orphan_tts", "stale_fit", "old_speaker"})


def test_workspace_skips_running_job(tmp_path, monkeypatch):
    work = tmp_path / "work"
    busy, idle = _job(work, "busy"), _job(work, "idle")
    monkeypatch.setattr(runlock, "_is_ours", lambda pid: True)
    (busy / runlock.LOCK_NAME).write_text(str(os.getppid()))
    ws = Workspace(tmp_path)
    assert {i.path.relative_to(work).parts[0] for i in ws.cleanup_plan()} == {"idle"}
    assert ws.cleanup() > 0
    assert (busy / "dub_mix_raw.wav").exists() and not (idle / "dub_mix_raw.wav").exists()


@pytest.mark.parametrize("n, text", [(0, "0 B"), (2048, "2.0 KB"), (5 * 1024**2, "5.0 MB"), (3 * 1024**3, "3.0 GB")])
def test_human(n, text):
    assert human(n) == text


def test_extra_keeps_source_videos_of_jobs(tmp_path):
    work = tmp_path / "work"
    wd = _job(work)
    _touch(work / "clips/source.mp4")
    _touch(work / "clips/unused.mp4")
    m = Manifest.load(wd / "manifest.json")
    m.source = str((work / "clips/source.mp4").resolve())
    m.save(wd / "manifest.json")
    _touch(work / "uploaded/job.json", json.dumps({"source": str(work / "raw.mov"), "output": "x"}).encode())
    _touch(work / "raw.mov")
    names = {i.path.name for i in plan(work, extra=True) if i.reason == "foreign"}
    assert "clips" not in names and "raw.mov" not in names
