"""[4] Időkeret-tudatos EN→HU fordítás LLM-mel."""

import json

from pydantic import BaseModel, create_model

from ..config import Config
from ..context import Context, StepError
from ..llm import LLMBackend, LLMError
from ..manifest import Manifest, Segment
from ..text import count_hu_syllables
from ..voice_kind import KIND_PROMPT

NAME = "translate"


def key(seg_id: int) -> str:
    return f"s{seg_id}"


def answer_schema(ids: list[int]) -> type[BaseModel]:
    """Szegmensenként egy kulcs ({"s12": "<magyar>"}): kb. 15%-kal kevesebb token, mint az id/text párok listája
    (a fordítás ideje a kimenet hosszával arányos), és a séma minden szegmensre kikényszeríti a választ."""
    return create_model("Answers", **{key(i): (str, ...) for i in ids})


def answers(batch: BaseModel) -> dict[int, str]:
    return {int(k[1:]): v.strip() for k, v in batch.model_dump(exclude_unset=True).items() if v and v.strip()}


def system_prompt(cfg: Config) -> str:
    address = (
        "Address the viewer informally (tegezés)."
        if cfg.formality == "tegezo"
        else "Address the viewer formally (magázás, Ön)."
    )
    glossary = ""
    if cfg.glossary:
        lines = "\n".join(f"- {en} → {hu}" for en, hu in cfg.glossary.items())
        glossary = f"\nMandatory glossary (always use these Hungarian terms):\n{lines}\n"
    return f"""You are a professional Hungarian dubbing translator. You translate English video transcript segments into natural, spoken Hungarian that a voice actor will read aloud over the original timing.

Rules:
- Idiomatic, fluent spoken Hungarian, never word-for-word. Keep the meaning and the tone.
- Every segment has a syllable budget `max_syl`. The Hungarian text MUST NOT exceed it. Shorter is fine; drop filler words and redundancy before dropping meaning.
- Translate each segment on its own: do not merge, split or move content between segments.
- `speaker` marks who is talking; keep each character's way of speaking consistent across the dialogue.
- When a speaker is a child, use simple, natural childlike Hungarian wording.
- {address}
- Write numbers, units, symbols and abbreviations as they are pronounced in Hungarian words (2025 → kétezer-huszonöt, 5% → öt százalék, AI → mesterséges intelligencia).
- Keep proper names unchanged.
{glossary}- Respond with JSON only."""


def speaking_rate(manifest: Manifest, speaker: str, cfg: Config) -> float:
    """A beszélő klónozott hangjának mért tempója (szótag/mp), ha van, különben a beállított érték."""
    sp = manifest.speakers.get(speaker)
    return sp.syllables_per_second if sp and sp.syllables_per_second else cfg.syllables_per_second


def speakers_note(manifest: Manifest) -> str:
    """A beszélők hangtípusa a promptba (pl. SPK0: adult woman), ha ismert."""
    kinds = {k: KIND_PROMPT[sp.kind] for k, sp in manifest.speakers.items() if sp.kind in KIND_PROMPT}
    return f"Speakers: {json.dumps(kinds)}\n\n" if kinds else ""


def syllable_budget(manifest: Manifest, index: int, cfg: Config) -> int:
    seg = manifest.segments[index]
    gap = max(manifest.next_start(index) - seg.end, 0.0)
    return max(round((seg.end - seg.start + 0.5 * gap) * speaking_rate(manifest, seg.speaker, cfg)), 3)


def summarize(llm: LLMBackend, segments: list[Segment]) -> str:
    transcript = " ".join(s.text_en for s in segments)
    return llm.complete_text(
        "You summarize video transcripts for a translator.",
        "In 3-5 English sentences describe the topic, the speaker(s), the tone and the key terms "
        f"of this video transcript:\n\n{transcript}",
    )


def translate_batch(ctx: Context, start: int, stop: int) -> None:
    cfg, m = ctx.cfg, ctx.manifest
    k = cfg.context_segments
    before = [
        {"id": key(s.id), "en": s.text_en, "hu": s.text_hu} for s in m.segments[max(start - k, 0) : start] if s.text_hu
    ]
    after = [{"id": key(s.id), "en": s.text_en} for s in m.segments[stop : stop + k]]
    pending = {i: m.segments[i] for i in range(start, stop)}

    for _ in range(3):
        todo = [
            {"id": key(s.id), "speaker": s.speaker, "en": s.text_en, "max_syl": syllable_budget(m, i, cfg)}
            for i, s in pending.items()
        ]
        user = (
            f"Video summary:\n{m.summary}\n\n"
            f"{speakers_note(m)}"
            f"Previous segments (already translated, for context only):\n{json.dumps(before, ensure_ascii=False)}\n\n"
            f"Segments to translate:\n{json.dumps(todo, ensure_ascii=False)}\n\n"
            f"Following segments (context only, do not translate):\n{json.dumps(after, ensure_ascii=False)}\n\n"
            'Return a JSON object that maps the id of each segment to translate to its Hungarian text: '
            '{"<id>": "<Hungarian>", ...}.'
        )
        schema = answer_schema([s.id for s in pending.values()])
        got = answers(ctx.llm.complete_json(system_prompt(cfg), user, schema))
        for i, s in list(pending.items()):
            if s.id in got:
                pending.pop(i).text_hu = got[s.id]
        if not pending:
            return
    raise StepError(f"Nem sikerült lefordítani ezeket a szegmenseket: {[s.id for s in pending.values()]}")


def shorten_batch(ctx: Context, requests: list[tuple[Segment, int]]) -> dict[int, str]:
    """Rövidebb magyar változatokat kér egyszerre több szegmenshez: {szegmens id: (seg, max. szótag)} → {id: szöveg}.

    Kötegelve hívjuk, hogy az LLM-et csak egyszer kelljen betölteni (a TTS-modell közben nincs a memóriában).
    """
    result: dict[int, str] = {}
    size = ctx.cfg.translate_batch_size
    for start in range(0, len(requests), size):
        chunk = requests[start : start + size]
        pending = {seg.id: (seg, target) for seg, target in chunk}
        for _ in range(3):
            items = [
                {
                    "id": key(seg.id),
                    "speaker": seg.speaker,
                    "en": seg.text_en,
                    "hu": seg.text_hu,
                    "syllables": count_hu_syllables(seg.text_hu or ""),
                    "max_syl": target,
                }
                for seg, target in pending.values()
            ]
            user = (
                f"{speakers_note(ctx.manifest)}"
                "These Hungarian dubbing lines are too long for their time slot. Rewrite each so it has at most "
                "`max_syl` syllables while keeping the essential meaning and the speaking style.\n\n"
                f"{json.dumps(items, ensure_ascii=False)}\n\n"
                'Return a JSON object that maps each id to the shorter Hungarian line: {"<id>": "<shorter Hungarian>", ...}.'
            )
            try:
                batch = ctx.shortener.complete_json(system_prompt(ctx.cfg), user, answer_schema(list(pending)))
            except LLMError:
                if not result:
                    raise
                return result  # a korábbi kötegek rövidítései megmaradnak
            for seg_id, text in answers(batch).items():
                if seg_id in pending:
                    result[seg_id] = text
                    pending.pop(seg_id)
            if not pending:
                break
    return result


def run(ctx: Context) -> None:
    m = ctx.manifest
    if not m.segments:
        raise StepError("Nincsenek szegmensek – előbb a transcribe lépésnek kell lefutnia.")
    ctx.console.print(f"  LLM: {ctx.cfg.llm}")
    if ctx.resuming and m.summary:
        done = sum(1 for s in m.segments if s.text_hu)
        ctx.console.print(f"  folytatás: {done}/{len(m.segments)} szegmens már le van fordítva")
    else:
        m.summary = summarize(ctx.llm, m.segments)
        for seg in m.segments:
            seg.text_hu = None
            seg.shorten_count = 0
            seg.edited = False
        ctx.save()

    size = ctx.cfg.translate_batch_size
    for start in range(0, len(m.segments), size):
        stop = min(start + size, len(m.segments))
        if all(s.text_hu for s in m.segments[start:stop]):
            continue
        ctx.console.print(f"  fordítás: {start + 1}–{stop} / {len(m.segments)}")
        translate_batch(ctx, start, stop)
        ctx.save()
    if hasattr(ctx.llm, "stats"):
        ctx.console.print(f"  LLM: {ctx.llm.stats()}")
