"""[6] Magyar beszédszintézis, rövidítési körökkel, ha egy mondat nem fér az időkeretbe.

Körönként: minden hiányzó szegmens legenerálódik (a túl hosszúak megjelölve); utána a TTS-modell kiürül,
egyetlen kötegelt LLM-hívás rövidíti a megjelölteket, majd csak azok generálódnak újra. Így a TTS és az LLM
soha nincs egyszerre a memóriában (16 GB-on ez okozta a swapolást és a lassú szegmenseket).
"""

import time
from pathlib import Path

from ..context import Context, StepError
from ..llm import LLMError
from ..manifest import Segment
from ..text import count_hu_syllables
from ..tts import TTSEngine
from .align import Fit, plan_fit
from .translate import shorten_batch

NAME = "synthesize"


def is_synthesized(seg: Segment, workdir: Path) -> bool:
    """Kész-e a szegmens hangja a jelenlegi szöveggel és beszélővel (folytatáskor átugorható)."""
    return (
        seg.action is not None
        and seg.tts_file is not None
        and seg.tts_text == seg.text_hu
        and seg.tts_speaker == seg.speaker
        and (workdir / seg.tts_file).exists()
    )


def shorten_target(text: str, duration: float, available: float) -> int:
    """Hány szótagra kell rövidíteni, hogy a kimondott szöveg beférjen (kis ráhagyással)."""
    return max(int(count_hu_syllables(text) * available / duration * 0.95), 3)


def load_tts(ctx: Context) -> TTSEngine:
    fresh = "tts" not in ctx.__dict__
    tts = ctx.tts
    if fresh:
        for name, speaker in ctx.manifest.speakers.items():
            if tts.clones_voice and not speaker.ref_audio:
                raise StepError(f"Nincs hangminta a(z) {name} beszélőhöz – futtasd újra a voices lépéstől.")
            ref = ctx.workdir / speaker.ref_audio if speaker.ref_audio else None
            tts.prepare_speaker(name, ref, speaker.kind)
            tts.set_rate(name, speaker.syllables_per_second)
    return tts


def synthesize_segment(ctx: Context, tts: TTSEngine, index: int, can_shorten: bool) -> tuple[Fit, float]:
    """Legenerálja a szegmenst; ha rövidíteni kell, Fit('shorten')-t ad vissza és nem véglegesít."""
    cfg, m = ctx.cfg, ctx.manifest
    seg = m.segments[index]
    out = ctx.path(f"tts/seg_{seg.id:04d}.wav")
    next_start = m.next_start(index)
    t0 = time.monotonic()

    dur = tts.synthesize(seg.text_hu, seg.speaker, out)
    attempts = tts.last_attempts
    fit = plan_fit(dur, seg.start, seg.end, next_start, cfg, can_shorten=can_shorten)

    speed, action = fit.speed, fit.action
    if fit.action != "shorten" and speed > cfg.resynth_min_speed and tts.supports_speed:
        # Nagyobb gyorsításnál a motor saját tempóváltása természetesebb, mint az utólagos nyújtás;
        # kis gyorsítást az align lépés nyújtással old meg (nem kell újra generálni).
        dur = tts.synthesize(seg.text_hu, seg.speaker, out, speed=speed)
        attempts += tts.last_attempts
        refit = plan_fit(dur, seg.start, seg.end, next_start, cfg, can_shorten=False)
        speed = refit.speed if refit.speed > 1.001 else 1.0
        if refit.action == "overflow":  # a motor tempóváltásával sem fér el
            action = "overflow"

    seg.synth_seconds = round((seg.synth_seconds or 0.0) + time.monotonic() - t0, 2)
    seg.tts_attempts = (seg.tts_attempts or 0) + attempts
    if fit.action == "shorten":
        seg.action, seg.tts_duration = None, round(dur, 3)
        return fit, dur

    seg.tts_file = str(out.relative_to(ctx.workdir))
    seg.tts_duration, seg.speed_factor, seg.action = round(dur, 3), round(speed, 4), action
    seg.tts_text, seg.tts_speaker = seg.text_hu, seg.speaker
    if action == "overflow":
        seg.note = "Maximális gyorsítással sem fér el, belelóg a következő szegmensbe."
    return fit, dur


def run(ctx: Context) -> None:
    cfg, m = ctx.cfg, ctx.manifest
    total = len(m.segments)
    for seg in m.segments:
        if not seg.text_hu:
            raise StepError(f"A(z) {seg.id}. szegmensnek nincs fordítása – előbb a translate lépés kell.")
    if ctx.resuming:
        todo = [i for i, s in enumerate(m.segments) if not is_synthesized(s, ctx.workdir)]
    else:
        # Új futás (pl. --from-step synthesize): minden szegmens hangja újra készül. A régi állapotot
        # azonnal töröljük, hogy egy megszakítás utáni folytatás ne keverje a régi és az új hangokat.
        todo = list(range(total))
        for seg in m.segments:
            seg.action, seg.tts_text = None, None
        ctx.save()
    if not todo:
        ctx.console.print("  minden szegmens kész")
        return
    if len(todo) < total:
        ctx.console.print(f"  folytatás: {total - len(todo)}/{total} szegmens már kész")
    for i in todo:
        seg = m.segments[i]
        seg.synth_seconds, seg.tts_attempts = None, None
        if seg.shorten_count == 0:
            seg.note = None

    round_no = 0
    while todo:
        if round_no:
            ctx.console.print(f"  {round_no}. rövidítési kör: {len(todo)} szegmens újragenerálása")
        tts = load_tts(ctx)
        too_long: list[tuple[Segment, int]] = []
        for i in todo:
            seg = m.segments[i]
            can_shorten = seg.shorten_count < cfg.shorten_rounds
            fit, dur = synthesize_segment(ctx, tts, i, can_shorten)
            if fit.action == "shorten":
                available = seg.end - seg.start + max(m.next_start(i) - seg.end, 0.0) * cfg.gap_borrow_ratio
                too_long.append((seg, shorten_target(seg.text_hu, dur, available)))
                label = "túl hosszú"
            else:
                label = fit.action
            ctx.console.print(f"  [{i + 1}/{total}] {seg.speaker} {label:<14} {seg.text_hu[:60]}")
            ctx.save()
        if not too_long:
            break

        # A TTS kiürül, mielőtt az LLM betöltődik – a kettő együtt nem fér el 16 GB-ban.
        ctx.release_models()
        ctx.console.print(f"  rövidítés: {len(too_long)} szegmens egy kötegben")
        try:
            shorter = shorten_batch(ctx, too_long)
        except LLMError as e:
            ctx.console.print(f"  [yellow]rövidítés nem sikerült ({e}); gyorsítás lesz helyette[/yellow]")
            shorter = {}
        finally:
            ctx.release_models()
        for seg, _ in too_long:
            if seg.id in shorter:
                seg.text_hu = shorter[seg.id]
                seg.shorten_count += 1
            else:
                # LLM nélkül is menjen tovább a szinkron: a következő körben gyorsítással illesztünk.
                seg.shorten_count = cfg.shorten_rounds
                seg.note = "Rövidítés helyett gyorsítva (az LLM nem volt elérhető)."
        ctx.save()
        index_of = {s.id: i for i, s in enumerate(m.segments)}
        todo = [index_of[seg.id] for seg, _ in too_long]
        round_no += 1
