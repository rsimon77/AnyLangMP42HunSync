"""[4] Időkeret-tudatos EN→HU fordítás LLM-mel."""

import json

from pydantic import BaseModel

from ..config import Config
from ..context import Context, StepError
from ..llm import LLMBackend
from ..manifest import Manifest, Segment
from ..text import count_hu_syllables
from ..voice_kind import KIND_PROMPT

NAME = "translate"


class Item(BaseModel):
    id: int
    text: str


class Batch(BaseModel):
    translations: list[Item]


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
        {"id": s.id, "en": s.text_en, "hu": s.text_hu} for s in m.segments[max(start - k, 0) : start] if s.text_hu
    ]
    after = [{"id": s.id, "en": s.text_en} for s in m.segments[stop : stop + k]]
    pending = {i: m.segments[i] for i in range(start, stop)}

    for _ in range(3):
        todo = [
            {"id": s.id, "speaker": s.speaker, "en": s.text_en, "max_syl": syllable_budget(m, i, cfg)}
            for i, s in pending.items()
        ]
        user = (
            f"Video summary:\n{m.summary}\n\n"
            f"{speakers_note(m)}"
            f"Previous segments (already translated, for context only):\n{json.dumps(before, ensure_ascii=False)}\n\n"
            f"Segments to translate:\n{json.dumps(todo, ensure_ascii=False)}\n\n"
            f"Following segments (context only, do not translate):\n{json.dumps(after, ensure_ascii=False)}\n\n"
            'Return {"translations": [{"id": <id>, "text": "<Hungarian>"}]} with exactly one item per segment to translate.'
        )
        batch = ctx.llm.complete_json(system_prompt(cfg), user, Batch)
        by_id = {s.id: i for i, s in pending.items()}
        for item in batch.translations:
            if item.id in by_id and item.text.strip():
                idx = by_id[item.id]
                pending.pop(idx).text_hu = item.text.strip()
                by_id.pop(item.id)
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
                    "id": seg.id,
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
                'Return {"translations": [{"id": <id>, "text": "<shorter Hungarian>"}]} with one item per line.'
            )
            batch = ctx.shortener.complete_json(system_prompt(ctx.cfg), user, Batch)
            for item in batch.translations:
                if item.id in pending and item.text.strip():
                    result[item.id] = item.text.strip()
                    pending.pop(item.id)
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
        ctx.save()

    size = ctx.cfg.translate_batch_size
    for start in range(0, len(m.segments), size):
        stop = min(start + size, len(m.segments))
        if all(s.text_hu for s in m.segments[start:stop]):
            continue
        ctx.console.print(f"  fordítás: {start + 1}–{stop} / {len(m.segments)}")
        translate_batch(ctx, start, stop)
        ctx.save()
