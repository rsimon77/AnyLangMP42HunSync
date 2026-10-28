"""Szegmensszerkesztő: a magyar szöveg és a beszélő javítása, majd csak a módosított mondatok újragenerálása.

A szerkesztés a manifestet írja. Az „alkalmazás” a futást a synthesize lépés közepére állítja vissza, így a
szokásos folytatás csak a nem kész szegmenseket generálja újra (a többi hangfájl marad), utána újra illeszt,
kever és kimuxol.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import time_stretch
from ..config import Config
from ..manifest import Manifest, Segment
from ..pipeline import STEP_NAMES
from ..steps.synthesize import is_synthesized
from ..voice_kind import classify_kind, pitch_correction

RESYNTH_FROM = "synthesize"
REMIX_FROM = "align"  # a hangmagasság az illesztéskor kerül a hangra: elég innen újrafuttatni
MAX_PITCH = 12.0  # félhang
PROBLEMS = ("forced_speedup", "overflow")  # ezeket érdemes meghallgatni


class EditError(ValueError):
    pass


def clock(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


def pending(m: Manifest, workdir: Path) -> list[Segment]:
    """A még (újra)generálandó szegmensek – szerkesztés vagy kért újragenerálás után."""
    return [s for s in m.segments if not is_synthesized(s, workdir)]


def status(seg: Segment, workdir: Path) -> str:
    """Rövid állapotjel a táblázatba: ✏️ újragenerálásra vár, ⚠️ gyorsítva / belelóg, ✓ rendben."""
    if not is_synthesized(seg, workdir):
        return "✏️"
    if seg.action in PROBLEMS:
        return "⚠️"
    return "✓"


def rows(m: Manifest, workdir: Path, only: str = "all") -> list[list]:
    """A táblázat sorai: #, idő, beszélő, állapot, angol, magyar. only: all | problems | pending."""
    out = []
    for s in m.segments:
        mark = status(s, workdir)
        if only == "problems" and mark != "⚠️":
            continue
        if only == "pending" and mark != "✏️":
            continue
        speed = f" {s.speed_factor:.2f}×" if s.speed_factor > 1.001 else ""
        pitch = f" {s.pitch_offset:+g}♪" if s.pitch_offset else ""
        out.append([s.id, f"{clock(s.start)}–{clock(s.end)}", s.speaker, mark + speed + pitch, s.text_en, s.text_hu or ""])
    return out


def find(m: Manifest, seg_id: int) -> Segment:
    for s in m.segments:
        if s.id == seg_id:
            return s
    raise EditError(f"Nincs ilyen szegmens: {seg_id}")


def edit(m: Manifest, seg_id: int, text_hu: str, speaker: str) -> bool:
    """Átírja a szegmenst; True, ha változott (akkor újragenerálás kell)."""
    seg = find(m, seg_id)
    text = " ".join(text_hu.split())
    if not text:
        raise EditError("A magyar szöveg nem lehet üres.")
    if m.speakers and speaker not in m.speakers:
        raise EditError(f"Ismeretlen beszélő: {speaker}")
    if text == seg.text_hu and speaker == seg.speaker:
        return False
    if text != seg.text_hu:
        seg.text_hu, seg.edited = text, True
    seg.speaker = speaker
    seg.note = None
    return True


def needs_remix(m: Manifest) -> bool:
    """Változott-e a hangmagasság az utolsó keverés óta (a videó elavult, de nem kell hangot generálni)."""
    before = STEP_NAMES[: STEP_NAMES.index(REMIX_FROM)]
    return set(before) <= set(m.steps_done) and STEP_NAMES[-1] not in m.steps_done


def invalidate_mix(m: Manifest) -> None:
    """Az illesztés, a keverés és a mux újrafut (a hangfájlok maradnak)."""
    keep = STEP_NAMES[: STEP_NAMES.index(REMIX_FROM)]
    m.steps_done = [s for s in m.steps_done if s in keep]
    if m.in_progress in STEP_NAMES[STEP_NAMES.index(REMIX_FROM) :]:
        m.in_progress = None


def check_pitch(semitones: float) -> float:
    st = round(float(semitones), 2)
    if abs(st) > MAX_PITCH:
        raise EditError(f"A hangmagasság legfeljebb ±{MAX_PITCH:g} félhang lehet.")
    return st


def set_line_pitch(m: Manifest, seg_id: int, semitones: float) -> bool:
    """Csak erre a mondatra: ennyi félhanggal magasabb (negatív: mélyebb) a beszélő hangjánál."""
    seg, st = find(m, seg_id), check_pitch(semitones)
    if st == seg.pitch_offset:
        return False
    seg.pitch_offset = st
    invalidate_mix(m)
    return True


def set_speaker_pitch(m: Manifest, speaker: str, semitones: float | None, cfg: Config) -> float:
    """A beszélő hangmagassága kézzel (félhang); None = vissza az automatikus korrekcióra. Az új értéket adja."""
    sp = m.speakers.get(speaker)
    if sp is None:
        raise EditError(f"Ismeretlen beszélő: {speaker}")
    if semitones is None:
        st, source = pitch_correction(sp.f0, sp.clone_f0, sp.kind, cfg), "auto"
    else:
        st, source = check_pitch(semitones), "config"
    if (st, source) != (sp.pitch_shift, sp.pitch_source):
        sp.pitch_shift, sp.pitch_source = st, source
        invalidate_mix(m)
    return st


def set_speaker_kind(m: Manifest, speaker: str, kind: str | None, cfg: Config) -> bool:
    """A beszélő hangtípusa kézzel (male/female/child), None = automatikus. Azonnal érvényes: a hangmagasság-
    korrekció újraszámolódik (ha nem kézi), Edge-hangnál a beszélő mondatai újragenerálódnak (más hang kell)."""
    sp = m.speakers.get(speaker)
    if sp is None:
        raise EditError(f"Ismeretlen beszélő: {speaker}")
    new = (classify_kind(sp.f0, cfg), "auto") if kind is None else (kind, "config")
    if new == (sp.kind, sp.kind_source):
        return False
    sp.kind, sp.kind_source = new
    if sp.pitch_source != "config":
        sp.pitch_shift, sp.pitch_source = pitch_correction(sp.f0, sp.clone_f0, sp.kind, cfg), "auto"
    if cfg.tts == "edge":  # az Edge a hangtípus szerint választ hangot
        for seg in m.segments:
            if seg.speaker == speaker:
                seg.tts_text = None
    invalidate_mix(m)
    return True


def edit_texts(m: Manifest, table: list[list]) -> list[int]:
    """A táblázatban átírt magyar szövegek mentése; a módosított mondatok azonosítóit adja."""
    changed = []
    for row in table:
        if not row or row[0] in (None, ""):
            continue
        seg = find(m, int(row[0]))
        text = " ".join(str(row[-1] or "").split())
        if text and text != (seg.text_hu or "") and edit(m, seg.id, text, seg.speaker):
            changed.append(seg.id)
    return changed


def preview(src: Path, out: Path, speed: float, semitones: float) -> str:
    """Meghallgatható előnézet: a hang a megadott gyorsítással és hangmagassággal (a végleges keverés nélkül)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    time_stretch(src, out, max(speed, 1.0), semitones)
    return str(out)


def line_preview(workdir: Path, m: Manifest, seg_id: int, semitones: float) -> str | None:
    """A mondat magyar hangja a beszélő hangmagasságával és a mondat saját eltolásával."""
    seg = find(m, seg_id)
    if not is_synthesized(seg, workdir):
        return None
    sp = m.speakers.get(seg.speaker)
    total = (sp.pitch_shift if sp else 0.0) + check_pitch(semitones)
    return preview(workdir / seg.tts_file, workdir / "preview" / f"seg_{seg.id:04d}.wav", seg.speed_factor, total)


def speaker_preview(workdir: Path, speaker: str, semitones: float) -> str | None:
    """A beszélő próbamondata (klónozott hang) a megadott hangmagassággal."""
    cal = workdir / "speakers" / f"{speaker}_cal0.wav"
    if not cal.exists():
        return None
    return preview(cal, workdir / "preview" / f"{speaker}.wav", 1.0, check_pitch(semitones))


def request_resynth(m: Manifest, seg_id: int) -> None:
    """Változatlan szöveggel is újragenerálja (pl. ha az XTTS elrontotta a kiejtést)."""
    find(m, seg_id).tts_text = None


def prepare_rerun(m: Manifest) -> None:
    """A futást a synthesize közepére állítja: a folytatás csak a nem kész szegmenseket generálja újra."""
    keep = STEP_NAMES[: STEP_NAMES.index(RESYNTH_FROM)]
    if not set(keep) <= set(m.steps_done):
        raise EditError("Előbb fusson le a szinkron legalább a fordításig.")
    m.steps_done = [s for s in m.steps_done if s in keep]
    m.in_progress = RESYNTH_FROM


def original_clip(workdir: Path, seg: Segment, pad: float = 0.3) -> tuple[int, np.ndarray] | None:
    """Az eredeti hang a szegmens idejéből (kis ráhagyással), a Gradio Audio formátumában."""
    src = workdir / "audio.wav"
    if not src.exists():
        return None
    info = sf.info(str(src))
    a = max(int((seg.start - pad) * info.samplerate), 0)
    b = min(int((seg.end + pad) * info.samplerate), info.frames)
    y, sr = sf.read(str(src), start=a, stop=b, dtype="float32", always_2d=True)
    return sr, y.mean(axis=1)


def dub_clip(workdir: Path, seg: Segment) -> str | None:
    """A szegmens magyar hangja úgy, ahogy a keverékbe kerül (gyorsítva / hangmagasság-korrekcióval, ha volt)."""
    if not is_synthesized(seg, workdir):
        return None
    raw, fit = workdir / seg.tts_file, workdir / f"tts/seg_{seg.id:04d}_fit.wav"
    # az illesztett változat csak akkor érvényes, ha az align a mostani hangfájlból készítette
    if fit.exists() and fit.stat().st_mtime >= raw.stat().st_mtime:
        return str(fit)
    return str(raw)
