"""A munkaterület (work/) takarítása: csak azt törli, amire egyik lépésnek vagy szerkesztésnek sincs szüksége.

Alap takarítás (mindig biztonságos, a munkák szerkeszthetők és folytathatók maradnak):
- előnézetek (preview/), félbemaradt írások (*.part.wav, *.tmp);
- dub_mix_raw.wav: a keverés köztes fájlja, csak a mix lépésen belül kell;
- olyan hangfájlok, amelyekre a manifest már nem hivatkozik (pl. újraátírás után), és elavult illesztett hangok;
- korábbi beszélő-csoportosítás hangmintái (a manifestben már nem szereplő beszélők);
- audio_16k.wav, ha megvan a tisztább asr_16k.wav (az átírás azt használja).

Extra (kérésre): a work/ alatti, munkához nem tartozó mappák és naplók, illetve a manifest régi másolatai.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .manifest import Manifest

KEEP_AT_ROOT = {"ollama.log"}  # a felület ide naplózza az általa indított Ollamát


@dataclass(frozen=True)
class Item:
    path: Path
    size: int
    reason: str  # preview | partial | mix_raw | orphan_tts | stale_fit | old_speaker | audio16k | backup | foreign


def size_of(p: Path) -> int:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size if p.exists() else 0


def is_job(d: Path) -> bool:
    return d.is_dir() and ((d / "manifest.json").exists() or (d / "job.json").exists())


def job_garbage(wd: Path) -> list[Item]:
    """Egy munkamappa haszontalan fájljai."""
    found: list[Item] = []

    def add(p: Path, reason: str) -> None:
        found.append(Item(p, size_of(p), reason))

    if (wd / "preview").is_dir():
        add(wd / "preview", "preview")
    for p in [*wd.rglob("*.part.wav"), *wd.glob("*.tmp")]:
        add(p, "partial")
    if (wd / "dub_mix_raw.wav").exists():
        add(wd / "dub_mix_raw.wav", "mix_raw")
    if (wd / "asr_16k.wav").exists() and (wd / "audio_16k.wav").exists():
        add(wd / "audio_16k.wav", "audio16k")
    for p in wd.glob("manifest.*.json"):
        add(p, "backup")

    try:
        m = Manifest.load(wd / "manifest.json") if (wd / "manifest.json").exists() else None
    except (ValueError, OSError):
        m = None  # sérült manifest: a hivatkozásokat nem ismerjük, ezért a hangfájlokhoz nem nyúlunk
    if m is None:
        return found
    used = {wd / s.tts_file for s in m.segments if s.tts_file}
    for p in sorted((wd / "tts").glob("seg_*.wav")):
        if p.name.endswith("_fit.wav"):
            raw = p.with_name(p.name.replace("_fit.wav", ".wav"))
            # az align mindig újra elkészíti; csak az árva és a forrásánál régebbi (elavult) változat felesleges
            if raw not in used or not raw.exists() or p.stat().st_mtime < raw.stat().st_mtime:
                add(p, "stale_fit")
        elif ".part" not in p.name and p not in used:
            add(p, "orphan_tts")
    if m.speakers:
        for p in sorted((wd / "speakers").glob("SPK*.wav")):
            if p.stem.split("_")[0] not in m.speakers:
                add(p, "old_speaker")
    return found


def job_sources(work: Path) -> set[Path]:
    """A munkák forrásvideói: ezek kellenek a mux lépéshez, akkor is, ha egy munkához nem tartozó mappában vannak."""
    sources = set()
    for d in work.iterdir():
        try:
            if (d / "job.json").exists():
                sources.add(Path(json.loads((d / "job.json").read_text(encoding="utf-8"))["source"]).resolve())
            if (d / "manifest.json").exists():
                sources.add(Path(Manifest.load(d / "manifest.json").source).resolve())
        except (ValueError, KeyError, OSError):
            continue
    return sources


def extra_garbage(work: Path) -> list[Item]:
    """A work/ alatt, munkához nem tartozó mappák és fájlok (pl. régi naplók, tesztmappák)."""
    sources = job_sources(work)
    found = []
    for p in sorted(work.iterdir()):
        if p.name.startswith(".") or p.name in KEEP_AT_ROOT or is_job(p):
            continue
        here = p.resolve()
        if any(src == here or here in src.parents for src in sources):
            continue  # egy munka forrásvideója (vagy azt tartalmazó mappa)
        found.append(Item(p, size_of(p), "foreign"))
    return found


def plan(work: Path, extra: bool = False, skip: set[Path] | None = None) -> list[Item]:
    """Mit törölne a takarítás; a skip munkamappáihoz (pl. éppen futó munka) nem nyúl."""
    if not work.exists():
        return []
    skip = {p.resolve() for p in skip or set()}
    items = []
    for d in sorted(work.iterdir()):
        if is_job(d) and d.resolve() not in skip:
            items += job_garbage(d)
    if extra:
        return items + extra_garbage(work)
    return [i for i in items if i.reason != "backup"]  # a manifest kézi másolatai csak az extra takarítással mennek


def remove(items: list[Item]) -> int:
    """Törli a tételeket; a felszabadított bájtok számát adja."""
    freed = 0
    for item in items:
        if item.path.is_dir():
            shutil.rmtree(item.path, ignore_errors=True)
        else:
            item.path.unlink(missing_ok=True)
        freed += item.size
    return freed


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"
