"""[3] Angol beszédfelismerés (MLX Whisper) és mondatszintű szegmentálás."""

from dataclasses import dataclass

from ..context import Context, StepError
from ..manifest import Segment

NAME = "transcribe"

SENTENCE_END = (".", "?", "!")
CLAUSE_END = (",", ";", ":", "—", "-")


@dataclass
class Word:
    text: str
    start: float
    end: float


def group_words(
    words: list[Word],
    max_dur: float = 12.0,
    pause_s: float = 0.7,
    min_dur: float = 1.0,
) -> list[tuple[float, float, str]]:
    """Szavakból szinkronozható szegmenseket épít.

    Mondatvégnél (ha már elég hosszú), hosszú szünetnél, illetve max_dur túllépésekor vág;
    utóbbi esetben lehetőleg az utolsó tagmondathatárnál.
    """
    segments: list[tuple[float, float, str]] = []
    cur: list[Word] = []

    def flush(upto: int | None = None) -> None:
        nonlocal cur
        part, rest = (cur, []) if upto is None else (cur[:upto], cur[upto:])
        text = "".join(w.text for w in part).strip()
        if text:
            segments.append((part[0].start, part[-1].end, text))
        cur = rest

    for w in words:
        if cur:
            if w.start - cur[-1].end >= pause_s:
                flush()
            elif w.end - cur[0].start > max_dur:
                cut = next(
                    (i + 1 for i in range(len(cur) - 1, 0, -1) if cur[i].text.strip().endswith(CLAUSE_END)),
                    None,
                )
                flush(cut)
        cur.append(w)
        if w.text.strip().endswith(SENTENCE_END) and cur[-1].end - cur[0].start >= min_dur:
            flush()
    if cur:
        flush()
    return segments


def run(ctx: Context) -> None:
    import mlx_whisper

    # A Demucs által leválasztott beszéd tisztább jel, kevesebb hallucinációval.
    source = ctx.path("asr_16k.wav")
    if not source.exists():
        source = ctx.path("audio_16k.wav")
    ctx.console.print(f"  Whisper modell: {ctx.cfg.whisper_model} ({source.name})")
    result = mlx_whisper.transcribe(
        str(source),
        path_or_hf_repo=ctx.cfg.whisper_model,
        language="en",
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    words = [
        Word(w["word"], float(w["start"]), float(w["end"]))
        for seg in result.get("segments", [])
        for w in seg.get("words", [])
    ]
    if not words:
        raise StepError("A Whisper nem talált beszédet a videóban.")

    groups = group_words(words)
    ctx.manifest.segments = [
        Segment(id=i, start=round(s, 3), end=round(e, 3), text_en=t) for i, (s, e, t) in enumerate(groups)
    ]
    ctx.console.print(f"  {len(words)} szó, {len(groups)} szegmens")
