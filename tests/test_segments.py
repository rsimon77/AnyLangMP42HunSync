import os

import numpy as np
import pytest
import soundfile as sf

from mp4translator import runlock
from mp4translator.manifest import Manifest, Segment, Speaker
from mp4translator.pipeline import STEP_NAMES
from mp4translator.webui import segments as seged
from mp4translator.webui.jobs import Workspace

DONE = STEP_NAMES[:]


def _manifest(tmp_path, n=3) -> Manifest:
    (tmp_path / "tts").mkdir(exist_ok=True)
    segs = []
    for i in range(n):
        f = f"tts/seg_{i:04d}.wav"
        sf.write(str(tmp_path / f), np.zeros(100, np.float32), 1000)
        segs.append(Segment(id=i, start=2.0 * i, end=2.0 * i + 1.5, text_en=f"line {i}", text_hu=f"sor {i}",
                            tts_file=f, tts_text=f"sor {i}", tts_speaker="SPK0", action="ok"))
    segs[2].action = "overflow"
    return Manifest(source="a.mp4", duration=6.0, segments=segs, steps_done=list(DONE),
                    speakers={"SPK0": Speaker(), "SPK1": Speaker()})


def test_rows_and_filters(tmp_path):
    m = _manifest(tmp_path)
    rows = seged.rows(m, tmp_path)
    assert [r[3] for r in rows] == ["✓", "✓", "⚠️"]
    assert rows[1][1] == "0:02.0–0:03.5"
    assert [r[0] for r in seged.rows(m, tmp_path, "problems")] == [2]
    assert seged.rows(m, tmp_path, "pending") == []


def test_edit_marks_line_for_resynth(tmp_path):
    m = _manifest(tmp_path)
    assert seged.edit(m, 1, "  javított   sor ", "SPK1")
    seg = m.segments[1]
    assert (seg.text_hu, seg.speaker, seg.edited) == ("javított sor", "SPK1", True)
    assert [s.id for s in seged.pending(m, tmp_path)] == [1]
    assert seged.rows(m, tmp_path)[1][3] == "✏️"
    assert seged.dub_clip(tmp_path, seg) is None  # a régi hang már nem érvényes


def test_speaker_only_change_keeps_text_unedited(tmp_path):
    m = _manifest(tmp_path)
    assert seged.edit(m, 0, "sor 0", "SPK1")
    assert not m.segments[0].edited and [s.id for s in seged.pending(m, tmp_path)] == [0]
    assert not seged.edit(m, 0, "sor 0", "SPK1")  # nincs változás


@pytest.mark.parametrize("text, speaker", [("   ", "SPK0"), ("jó", "SPK9")])
def test_edit_rejects_bad_input(tmp_path, text, speaker):
    with pytest.raises(seged.EditError):
        seged.edit(_manifest(tmp_path), 0, text, speaker)


def test_request_resynth_with_same_text(tmp_path):
    m = _manifest(tmp_path)
    seged.request_resynth(m, 2)
    assert [s.id for s in seged.pending(m, tmp_path)] == [2]
    with pytest.raises(seged.EditError):
        seged.request_resynth(m, 99)


def test_prepare_rerun_resumes_synthesize(tmp_path):
    m = _manifest(tmp_path)
    seged.prepare_rerun(m)
    assert m.steps_done == STEP_NAMES[: STEP_NAMES.index("synthesize")]
    assert m.in_progress == "synthesize"
    with pytest.raises(seged.EditError):
        seged.prepare_rerun(Manifest(source="a.mp4", steps_done=["extract"]))


def test_audio_clips(tmp_path):
    m = _manifest(tmp_path)
    sf.write(str(tmp_path / "audio.wav"), np.zeros((8000, 2), np.float32), 1000)
    sr, y = seged.original_clip(tmp_path, m.segments[1])
    assert sr == 1000 and y.size == 2100  # 1,5 mp + 2 × 0,3 mp ráhagyás
    seg = m.segments[0]
    assert seged.dub_clip(tmp_path, seg).endswith("seg_0000.wav")
    fit = tmp_path / "tts/seg_0000_fit.wav"
    sf.write(str(fit), np.zeros(80, np.float32), 1000)
    assert seged.dub_clip(tmp_path, seg) == str(fit)
    os.utime(fit, (0, 0))  # régebbi, mint az újragenerált hang → elavult
    assert seged.dub_clip(tmp_path, seg).endswith("seg_0000.wav")


@pytest.fixture
def job(tmp_path):
    ws = Workspace(tmp_path)
    wd = tmp_path / "work" / "clip"
    wd.mkdir(parents=True)
    _manifest(wd).save(wd / "manifest.json")
    return ws, ws.load("clip")


def test_workspace_edit_refuses_while_running(job, monkeypatch):
    ws, j = job
    ws.edit_manifest(j, lambda m: seged.edit(m, 0, "új", "SPK0"))
    assert j.manifest().segments[0].text_hu == "új"
    monkeypatch.setattr(runlock, "_is_ours", lambda pid: True)
    (j.workdir / runlock.LOCK_NAME).write_text(str(os.getppid()))
    with pytest.raises(RuntimeError):
        ws.edit_manifest(j, lambda m: seged.edit(m, 0, "más", "SPK0"))


def test_rerun_pending_resets_and_starts(job, monkeypatch):
    ws, j = job
    started = []
    monkeypatch.setattr(ws, "start", lambda job, from_step=None: started.append(from_step))
    ws.rerun_pending(j)
    m = j.manifest()
    assert m.in_progress == "synthesize" and "align" not in m.steps_done
    assert started == [None]  # sima folytatás, nem --from-step (az mindent újragenerálna)


def test_line_pitch_needs_only_remix(tmp_path):
    m = _manifest(tmp_path)
    assert not seged.needs_remix(m)
    assert seged.set_line_pitch(m, 1, 2.5)
    assert m.segments[1].pitch_offset == 2.5
    assert seged.needs_remix(m) and seged.pending(m, tmp_path) == []  # nem kell új hang
    assert "synthesize" in m.steps_done and "align" not in m.steps_done
    assert "+2.5♪" in seged.rows(m, tmp_path)[1][3]
    assert not seged.set_line_pitch(m, 1, 2.5)
    with pytest.raises(seged.EditError):
        seged.set_line_pitch(m, 1, 13)


def test_speaker_pitch_manual_and_auto(tmp_path):
    from mp4translator.config import Config

    m = _manifest(tmp_path)
    sp = m.speakers["SPK0"]
    sp.f0, sp.clone_f0, sp.kind = 400.0, 300.0, "child"
    cfg = Config()
    assert seged.set_speaker_pitch(m, "SPK0", -1.5, cfg) == -1.5
    assert (sp.pitch_shift, sp.pitch_source) == (-1.5, "config") and seged.needs_remix(m)
    auto = seged.set_speaker_pitch(m, "SPK0", None, cfg)  # vissza az automatikus korrekcióra
    assert sp.pitch_source == "auto" and auto == pytest.approx(4.98, abs=0.01)
    with pytest.raises(seged.EditError):
        seged.set_speaker_pitch(m, "SPK9", 1, cfg)


def test_rerun_after_pitch_only_change_skips_synthesis(job, monkeypatch):
    ws, j = job
    ws.edit_manifest(j, lambda m: seged.set_line_pitch(m, 0, 1.0))
    monkeypatch.setattr(ws, "start", lambda job, from_step=None: None)
    ws.rerun_pending(j)
    m = j.manifest()
    assert m.in_progress == "synthesize" and seged.pending(m, j.workdir) == []  # a synthesize azonnal kész


def test_previews_shift_pitch(tmp_path):
    m = _manifest(tmp_path)
    sr = 16000
    tone = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32) * 0.5
    sf.write(str(tmp_path / m.segments[0].tts_file), tone, sr)
    out = seged.line_preview(tmp_path, m, 0, 12)
    y, _ = sf.read(out)
    spectrum = np.abs(np.fft.rfft(y))
    assert abs(np.argmax(spectrum) * sr / (2 * (spectrum.size - 1)) - 440) < 10  # egy oktávval feljebb
    (tmp_path / "speakers").mkdir()
    sf.write(str(tmp_path / "speakers/SPK0_cal0.wav"), tone, sr)
    assert seged.speaker_preview(tmp_path, "SPK0", -3).endswith("preview/SPK0.wav")
    assert seged.speaker_preview(tmp_path, "SPK1", 1) is None
