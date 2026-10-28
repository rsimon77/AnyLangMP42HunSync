"""Szegmensszerkesztő: a magyar szöveg és a beszélő javítása, majd csak a módosított mondatok újragenerálása.

A szerkesztés a manifestet írja. Az „alkalmazás” a futást a synthesize lépés közepére állítja vissza, így a
szokásos folytatás csak a nem kész szegmenseket generálja újra (a többi hangfájl marad), utána újra illeszt,
kever és kimuxol.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from ..manifest import Manifest, Segment
from ..pipeline import STEP_NAMES
from ..steps.synthesize import is_synthesized

RESYNTH_FROM = "synthesize"
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
        out.append([s.id, f"{clock(s.start)}–{clock(s.end)}", s.speaker, mark + speed, s.text_en, s.text_hu or ""])
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
