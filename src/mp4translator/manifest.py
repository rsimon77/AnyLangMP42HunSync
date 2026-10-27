from pathlib import Path

from pydantic import BaseModel


class Segment(BaseModel):
    id: int
    start: float
    end: float
    speaker: str = "SPK0"
    text_en: str
    text_hu: str | None = None
    tts_file: str | None = None
    tts_duration: float | None = None
    tts_text: str | None = None  # a hangfájlban elhangzó szöveg; ha eltér a text_hu-tól, újragenerálás kell
    tts_speaker: str | None = None
    speed_factor: float = 1.0
    action: str | None = None  # ok | borrow | speedup | forced_speedup | overflow
    shorten_count: int = 0
    note: str | None = None
    synth_seconds: float | None = None  # a hanggenerálásra fordított idő (méréshez)
    tts_attempts: int | None = None  # TTS-hívások száma, újrapróbálásokkal együtt


class Speaker(BaseModel):
    ref_audio: str | None = None  # a munkamappához képest
    seconds: float = 0.0
    syllables_per_second: float | None = None  # a klónozott hang mért tempója
    kind: str | None = None  # male | female | child
    kind_source: str | None = None  # auto | config
    f0: float | None = None  # az eredeti hang alaphangjának mediánja (Hz)
    clone_f0: float | None = None  # a klónozott hangé (Hz)
    pitch_shift: float = 0.0  # félhang; az align lépés alkalmazza


class KeptSound(BaseModel):
    """Az énekhang-sávból megtartott nem-beszéd hang (pl. állathang) – változatlanul a háttérbe kerül."""

    start: float
    end: float
    label: str
    score: float


class Manifest(BaseModel):
    source: str
    duration: float = 0.0
    summary: str | None = None
    steps_done: list[str] = []
    in_progress: str | None = None  # a megszakadt lépés neve; ilyenkor folytatás, nem újrakezdés
    speakers: dict[str, Speaker] = {}
    segments: list[Segment] = []
    kept_sounds: list[KeptSound] = []

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    def next_start(self, index: int) -> float:
        """A következő szegmens kezdete, az utolsónál a videó vége."""
        if index + 1 < len(self.segments):
            return self.segments[index + 1].start
        return self.duration
