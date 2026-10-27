from pathlib import Path

import pytest
from rich.console import Console

from mp4translator import context as context_mod
from mp4translator.config import Config
from mp4translator.context import Context
from mp4translator.llm import LLMError
from mp4translator.manifest import Manifest, Segment, Speaker
from mp4translator.steps import synthesize
from mp4translator.steps.translate import Batch, Item, shorten_batch, speaking_rate, syllable_budget
from mp4translator.steps.voices import clamp_rate, measure_rate
from mp4translator.text import count_hu_syllables

RATE = 6.0  # az ál-TTS tempója, szótag/mp


class FakeTTS:
    supports_speed = True
    clones_voice = True
    live = 0  # egyszerre betöltött példányok száma

    def __init__(self, log):
        self.log = log
        self.last_attempts = 1
        self.prepared = set()
        self.rates = {}
        FakeTTS.live += 1

    def prepare_speaker(self, speaker, ref_audio, kind=None):
        self.prepared.add(speaker)

    def set_rate(self, speaker, syllables_per_second):
        self.rates[speaker] = syllables_per_second

    def synthesize(self, text, speaker, out, speed=1.0):
        assert speaker in self.prepared
        self.log.append(("tts", text, speed))
        Path(out).write_bytes(b"x")
        return count_hu_syllables(text) / RATE / speed

    def close(self):
        FakeTTS.live -= 1


class FakeLLM:
    def __init__(self, log, answers=None, error=False):
        self.log, self.answers, self.error = log, answers or {}, error

    def complete_json(self, system, user, schema):
        assert FakeTTS.live == 0, "az LLM a TTS-sel egyszerre lenne a memóriában"
        self.log.append(("llm", user))
        if self.error:
            raise LLMError("nincs Ollama")
        return Batch(translations=[Item(id=k, text=v) for k, v in self.answers.items()])

    def unload(self):
        self.log.append(("unload",))


@pytest.fixture
def make_ctx(tmp_path, monkeypatch):
    FakeTTS.live = 0
    log = []

    def build(texts, slot=2.0, gap=0.0, answers=None, error=False, **cfg):
        segs = []
        t = 0.0
        for i, text in enumerate(texts):
            segs.append(Segment(id=i, start=t, end=t + slot, text_en="x", text_hu=text))
            t += slot + gap
        m = Manifest(source="a.mp4", duration=t, segments=segs, speakers={"SPK0": Speaker(ref_audio="ref.wav")})
        llm = FakeLLM(log, answers, error)
        monkeypatch.setattr(context_mod, "make_tts", lambda c: FakeTTS(log))
        monkeypatch.setattr(context_mod, "make_llm", lambda c, spec=None: llm)
        config = Config(**{"shorten_rounds": 2, "resynth_min_speed": 1.10, **cfg})
        return Context(config, Path("in.mp4"), Path("out.mp4"), tmp_path, m, console=Console(quiet=True))

    return build, log


SHORT = "Szia."  # 2 szótag, 0,33 mp
LONG = "Ez egy nagyon hosszú mondat, ami semmiképp sem fér bele a rendelkezésre álló időbe."  # 2 mp-be nem fér


def test_fitting_segments_need_no_llm(make_ctx):
    build, log = make_ctx
    ctx = build([SHORT, SHORT])
    synthesize.run(ctx)
    assert all(s.action == "ok" and s.tts_text == SHORT for s in ctx.manifest.segments)
    assert not any(e[0] == "llm" for e in log)


def test_too_long_segment_is_shortened_after_tts_is_released(make_ctx):
    build, log = make_ctx
    ctx = build([SHORT, LONG], answers={1: "Ez túl hosszú."})
    synthesize.run(ctx)
    seg = ctx.manifest.segments[1]
    assert seg.text_hu == "Ez túl hosszú." and seg.tts_text == seg.text_hu
    assert seg.shorten_count == 1 and seg.action == "ok"
    # sorrend: két TTS, egy LLM-köteg, majd csak a rövidített szegmens generálódik újra
    kinds = [(e[0], e[1] if e[0] == "tts" else None) for e in log if e[0] != "unload"]
    assert kinds == [("tts", SHORT), ("tts", LONG), ("llm", None), ("tts", "Ez túl hosszú.")]
    assert ctx.manifest.segments[0].tts_attempts == 1


def test_all_too_long_segments_go_in_one_llm_call(make_ctx):
    build, log = make_ctx
    ctx = build([LONG, LONG, LONG], answers={0: SHORT, 1: SHORT, 2: SHORT})
    synthesize.run(ctx)
    assert sum(e[0] == "llm" for e in log) == 1
    assert all(s.text_hu == SHORT for s in ctx.manifest.segments)


def test_llm_error_falls_back_to_speedup(make_ctx):
    build, log = make_ctx
    ctx = build([LONG], error=True)
    synthesize.run(ctx)
    seg = ctx.manifest.segments[0]
    assert seg.text_hu == LONG
    assert seg.shorten_count == ctx.cfg.shorten_rounds
    assert seg.action in ("forced_speedup", "overflow")
    assert seg.note


def test_rounds_are_bounded_when_llm_never_shortens_enough(make_ctx):
    build, log = make_ctx
    ctx = build([LONG], answers={0: LONG})
    synthesize.run(ctx)
    assert sum(e[0] == "llm" for e in log) == ctx.cfg.shorten_rounds
    assert ctx.manifest.segments[0].action in ("forced_speedup", "overflow")


def test_large_speedup_resynthesizes_with_engine_speed(make_ctx):
    build, log = make_ctx
    # 10 szótag = 1,67 mp egy 1,5 mp-es helyen: 1,11×
    ctx = build(["Holnap reggel elmegyünk a boltba."], slot=1.5, soft_speedup=1.15, resynth_min_speed=1.05)
    synthesize.run(ctx)
    speeds = [e[2] for e in log if e[0] == "tts"]
    assert len(speeds) == 2 and speeds[1] > 1.05
    assert ctx.manifest.segments[0].speed_factor == 1.0


def test_small_speedup_is_left_to_time_stretch(make_ctx):
    build, log = make_ctx
    ctx = build(["Holnap reggel elmegyünk a boltba."], slot=1.5, soft_speedup=1.15, resynth_min_speed=1.12)
    synthesize.run(ctx)
    assert [e[2] for e in log if e[0] == "tts"] == [1.0]
    seg = ctx.manifest.segments[0]
    assert seg.action == "speedup" and 1.0 < seg.speed_factor <= 1.12


def test_resume_skips_finished_segments(make_ctx):
    build, log = make_ctx
    ctx = build([SHORT, SHORT])
    synthesize.run(ctx)
    log.clear()
    ctx.manifest.segments[1].text_hu = "Helló."
    ctx.resuming = True
    synthesize.run(ctx)
    assert [e[1] for e in log if e[0] == "tts"] == ["Helló."]


def test_fresh_run_regenerates_everything(make_ctx):
    build, log = make_ctx
    ctx = build([SHORT, SHORT])
    synthesize.run(ctx)
    log.clear()
    synthesize.run(ctx)  # pl. --from-step synthesize
    assert sum(e[0] == "tts" for e in log) == 2


def test_shorten_target_scales_with_available_time():
    # 20 szótag 4 mp alatt, 2 mp hely → kb. 9-10 szótag
    text = "a" * 20
    assert synthesize.shorten_target(text, 4.0, 2.0) == 9
    assert synthesize.shorten_target(text, 4.0, 0.1) == 3


def test_shorten_batch_retries_missing_items(make_ctx):
    build, log = make_ctx
    ctx = build([LONG, LONG])
    calls = iter([Batch(translations=[Item(id=0, text="Első.")]), Batch(translations=[Item(id=1, text="Második.")])])
    ctx.__dict__["shortener"] = type("L", (), {"complete_json": lambda self, s, u, schema: next(calls)})()
    segs = ctx.manifest.segments
    assert shorten_batch(ctx, [(segs[0], 5), (segs[1], 5)]) == {0: "Első.", 1: "Második."}


def test_clamp_rate():
    assert clamp_rate(30, 5.0) == 6.0
    assert clamp_rate(30, 1.0) == 9.0
    assert clamp_rate(10, 10.0) == 4.0
    assert clamp_rate(0, 1.0) is None
    assert clamp_rate(10, 0.0) is None


def test_measure_rate_uses_cloned_voice(make_ctx):
    build, _ = make_ctx
    ctx = build([SHORT])
    ctx.tts.prepare_speaker("SPK0", None)
    assert measure_rate(ctx, "SPK0", ctx.workdir) == RATE
    assert (ctx.workdir / "SPK0_cal0.wav").exists()  # a próbamondatok megmaradnak (hangmagasság-méréshez)


def test_budget_uses_measured_speaker_rate():
    m = Manifest(
        source="a.mp4",
        duration=10,
        segments=[Segment(id=0, start=0, end=2, text_en="x", speaker="SPK1")],
        speakers={"SPK1": Speaker(syllables_per_second=8.0), "SPK0": Speaker()},
    )
    cfg = Config(syllables_per_second=6.0)
    assert speaking_rate(m, "SPK1", cfg) == 8.0
    assert speaking_rate(m, "SPK0", cfg) == 6.0
    assert syllable_budget(m, 0, cfg) == round((2 + 0.5 * 8) * 8)


def test_reloaded_tts_gets_measured_speaker_rates(make_ctx):
    build, _ = make_ctx
    ctx = build([SHORT])
    ctx.manifest.speakers["SPK0"].syllables_per_second = 4.4
    synthesize.run(ctx)
    assert ctx.tts.rates == {"SPK0": 4.4}
