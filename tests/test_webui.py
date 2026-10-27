import os
import sys
from pathlib import Path

import pytest
import yaml

from mp4translator import runlock
from mp4translator.manifest import Manifest
from mp4translator.webui import jobs as jobs_mod
from mp4translator.webui.i18n import TEXTS, pick_lang, t
from mp4translator.webui.jobs import Workspace, slugify, step_states, tail, unique_name


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "config.yaml").write_text("llm: ollama:gemma3:12b\nformality: tegezo\n")
    return Workspace(tmp_path)


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "Barbie Family Story – Chelsea!.mp4"
    p.write_bytes(b"fake")
    return p


def test_texts_have_same_keys():
    assert set(TEXTS["en"]) == set(TEXTS["hu"])


def test_pick_lang():
    assert pick_lang("hu-HU,hu;q=0.9,en;q=0.8") == "hu"
    assert pick_lang("de-DE,en;q=0.5,hu;q=0.4") == "en"
    assert pick_lang(None) == "en"
    assert t("hu", "created", name="x") == "Új munka létrehozva: x"


def test_slugify_and_unique_name():
    assert slugify("Barbie Family Story – Chelsea!") == "barbie_family_story_chelsea"
    assert slugify("Árvíztűrő tükörfúrógép") == "arvizturo_tukorfurogep"
    assert slugify("???") == "video"
    assert unique_name("a", {"a", "a_2"}) == "a_3"


def test_create_job_copies_video(ws, video):
    job = ws.create(video)
    assert job.name == "barbie_family_story_chelsea"
    assert job.source.read_bytes() == b"fake" and job.source.parent == job.workdir
    assert job.output == ws.root / "output" / "barbie_family_story_chelsea_hu.mp4"
    assert ws.load(job.name) == job
    assert ws.create(video).name == "barbie_family_story_chelsea_2"
    assert [j.name for j in ws.jobs()] == ["barbie_family_story_chelsea", "barbie_family_story_chelsea_2"]


def test_create_rejects_non_video(ws, tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("x")
    with pytest.raises(ValueError):
        ws.create(p)


def test_cli_job_without_job_file_is_listed(ws):
    wd = ws.work / "barbie"
    wd.mkdir(parents=True)
    Manifest(source="/videos/barbie.mp4").save(wd / "manifest.json")
    job = ws.load("barbie")
    assert job.source == Path("/videos/barbie.mp4")
    assert job.output == ws.root / "output" / "barbie_hu.mp4"


def test_save_config_merges_and_validates(ws, video):
    job = ws.create(video)
    cfg = ws.save_config(job, {"formality": "magazo", "glossary": {"AI": "MI"}})
    assert cfg.formality == "magazo" and cfg.llm == "ollama:gemma3:12b"
    ws.save_config(job, {"speaker_kinds": {"SPK1": "child"}})
    saved = yaml.safe_load(job.config_path.read_text())
    assert saved["formality"] == "magazo" and saved["speaker_kinds"] == {"SPK1": "child"}
    with pytest.raises(Exception):
        ws.save_config(job, {"tts": "nonexistent"})
    assert ws.job_config(job).tts == "xtts"  # a hibás érték nem íródott ki


def test_command(ws, video):
    job = ws.create(video)
    cmd = ws.command(job)
    assert cmd[:4] == [sys.executable, "-m", "mp4translator.cli", "dub"]
    assert cmd[cmd.index("-c") + 1] == str(ws.root / "config.yaml")
    ws.save_config(job, {})
    cmd = ws.command(job, "voices")
    assert cmd[cmd.index("-c") + 1] == str(job.config_path)
    assert cmd[-2:] == ["--from-step", "voices"]
    with pytest.raises(ValueError):
        ws.command(job, "nope")


def test_only_one_run_at_a_time(ws, video):
    job = ws.create(video)
    assert ws.running() is None
    (job.workdir / runlock.LOCK_NAME).write_text(str(os.getppid()))  # élő, másik folyamat
    assert ws.running() == job
    with pytest.raises(RuntimeError):
        ws.start(ws.create(video))
    (job.workdir / runlock.LOCK_NAME).write_text("999999")  # elárvult zár
    assert ws.running() is None


def test_start_runs_cli_in_background(ws, video, monkeypatch):
    job = ws.create(video)
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            calls.append((cmd, kw))

        def poll(self):
            return None

    monkeypatch.setattr(jobs_mod.subprocess, "Popen", FakePopen)
    ws.start(job, "translate")
    cmd, kw = calls[0]
    assert cmd[-1] == "translate" and kw["cwd"] == ws.root and kw["start_new_session"]
    assert ws.running() == job  # a zárfájl megjelenése előtt is foglalt


def test_step_states():
    m = Manifest(source="a.mp4", steps_done=["extract", "separate"], in_progress="transcribe")
    states = dict(step_states(m, running=True))
    assert states["extract"] == "done" and states["transcribe"] == "running" and states["mux"] == "pending"
    assert dict(step_states(m, running=False))["transcribe"] == "paused"
    assert all(s == "pending" for _, s in step_states(None, False))


def test_tail_drops_library_noise(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("▶ voices\nfoo.py:1: UserWarning: bla\n  warnings.warn(\n  kész (3 mp)\n")
    assert tail(log) == "▶ voices\n  kész (3 mp)"
    assert tail(tmp_path / "missing.log") == ""


def test_app_builds(ws):
    from mp4translator.webui.app import build

    assert build(ws) is not None


def test_tail_strips_colors_and_progress_redraws(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\x1b[1;36m▶ separate\x1b[0m\n  0%|  |\r 50%|##|\r100%|####|\n\n  kész\n")
    assert tail(log) == "▶ separate\n100%||\n  kész"
