from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    max_duration_s: float = 600.0

    whisper_model: str = "mlx-community/whisper-large-v3-turbo"

    llm: str = "ollama:gemma3:12b"
    ollama_host: str = "http://localhost:11434"
    llm_num_ctx: int = 4096  # kisebb kontextus → kevesebb memória (egy 10 perces átirat kb. 2000 token)
    shorten_llm: str | None = None  # külön (kisebb) modell a rövidítéshez, pl. ollama:gemma3:4b
    translate_batch_size: int = 20
    context_segments: int = 3
    formality: Literal["tegezo", "magazo"] = "tegezo"
    syllables_per_second: float = 6.0

    torch_device: str = "auto"  # auto | mps | cpu
    demucs_model: str = "htdemucs"

    tts: Literal["xtts", "edge"] = "xtts"
    edge_voice: str = "hu-HU-TamasNeural"  # ha a hangtípus ismeretlen
    edge_voices: dict[str, str] = {
        "male": "hu-HU-TamasNeural",
        "female": "hu-HU-NoemiNeural",
        "child": "hu-HU-NoemiNeural",
    }
    edge_child_pitch: str = "+25Hz"  # gyerekhangnál ennyivel magasabb
    # Az XTTS-v2 a Coqui Public Model License alatt érhető el (csak nem kereskedelmi célra);
    # a felhasználónak kell elfogadnia a config.yaml-ban.
    xtts_accept_license: bool = False
    xtts_temperature: float = 0.65
    xtts_retries: int = 3

    diarize: bool = True
    speaker_threshold: float = 0.35  # koszinusz-távolság a beszélőcsoportok összevonásához
    min_speaker_seconds: float = 4.0  # ennél kevesebb beszédű csoport a legközelebbihez olvad
    ref_audio_seconds: float = 12.0

    # Hangtípus (male | female | child) az alaphang mediánja alapján; kézzel felülírható beszélőnként
    male_max_f0: float = 165.0  # Hz; ez alatt férfi
    child_min_f0: float = 300.0  # Hz; e fölött gyerek
    speaker_kinds: dict[str, Literal["male", "female", "child"]] = {}  # pl. {SPK2: child}
    # A klónozott hang gyakran mélyebb az eredetinél (gyerekhangból felnőttes lesz); ennyit korrigálunk
    pitch_correction: Literal["off", "child", "all"] = "child"
    min_pitch_shift: float = 1.5  # félhang; ennél kisebb eltérést nem korrigálunk
    max_pitch_shift: float = 7.0  # félhang; felső határ (nagyobb eltolás már torzít)
    # Kézi hangmagasság beszélőnként (félhang), felülírja az automatikus korrekciót; pl. {SPK1: 2.5}
    speaker_pitch: dict[str, Annotated[float, Field(ge=-12, le=12)]] = {}

    # Állathangok (ugatás, nyávogás stb.) megtartása a beszédszünetekből
    keep_animal_sounds: bool = True
    animal_threshold: float = 0.3  # AudioSet-valószínűség (AST-modell)

    gap_borrow_ratio: float = 0.8
    soft_speedup: float = 1.15
    max_speedup: float = 1.3
    shorten_rounds: int = 2
    resynth_min_speed: float = 1.10  # eddig a gyorsításig nyújtás, efölött újragenerálás gyorsabb tempóval

    sample_rate: int = 44100
    background: Literal["separate", "duck", "none"] = "separate"
    duck_db: float = -22.0
    keep_original_audio: bool = True
    write_srt: bool = True

    glossary: dict[str, str] = {}

    @classmethod
    def load(cls, path: Path | None) -> "Config":
        if path is None or not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)
