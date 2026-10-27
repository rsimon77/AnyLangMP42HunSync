from mp4translator.config import Config
from mp4translator.manifest import Manifest, Segment
from mp4translator.steps.translate import syllable_budget, system_prompt


def manifest(*spans, duration=20.0):
    segs = [Segment(id=i, start=s, end=e, text_en="x") for i, (s, e) in enumerate(spans)]
    return Manifest(source="a.mp4", duration=duration, segments=segs)


def test_budget_includes_half_gap():
    cfg = Config(syllables_per_second=6.0)
    m = manifest((0, 2), (4, 6))
    # (2 + 0.5*2) * 6 = 18
    assert syllable_budget(m, 0, cfg) == 18


def test_budget_last_segment_uses_video_end():
    cfg = Config(syllables_per_second=6.0)
    m = manifest((0, 2), (4, 6), duration=8.0)
    assert syllable_budget(m, 1, cfg) == round((2 + 0.5 * 2) * 6)


def test_budget_negative_gap_ignored():
    cfg = Config(syllables_per_second=6.0)
    m = manifest((0, 2), (1.5, 3))
    assert syllable_budget(m, 0, cfg) == 12


def test_budget_minimum_three():
    m = manifest((0, 0.1), (0.1, 0.2))
    assert syllable_budget(m, 0, Config()) == 3


def test_prompt_glossary():
    p = system_prompt(Config(glossary={"pipeline": "adatcsővezeték", "cloud": "felhő"}))
    assert "pipeline → adatcsővezeték" in p
    assert "cloud → felhő" in p
    assert "glossary" in p


def test_prompt_without_glossary():
    assert "glossary" not in system_prompt(Config())


def test_prompt_formality():
    informal = system_prompt(Config(formality="tegezo"))
    formal = system_prompt(Config(formality="magazo"))
    assert "informally" in informal and "formally (magázás" in formal
    assert "informally" not in formal
    assert "magázás" not in informal
