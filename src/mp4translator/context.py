from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from rich.console import Console

from .config import Config
from .llm import LLMBackend, make_llm
from .manifest import Manifest
from .tts import TTSEngine, make_tts


class StepError(RuntimeError):
    pass


@dataclass
class Context:
    cfg: Config
    input_path: Path
    output_path: Path
    workdir: Path
    manifest: Manifest
    console: Console = field(default_factory=Console)
    resuming: bool = False  # az aktuális lépés egy megszakított futást folytat

    def path(self, name: str) -> Path:
        p = self.workdir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def manifest_path(self) -> Path:
        return self.workdir / "manifest.json"

    def save(self) -> None:
        self.manifest.save(self.manifest_path)

    @cached_property
    def llm(self) -> LLMBackend:
        return make_llm(self.cfg)

    @cached_property
    def shortener(self) -> LLMBackend:
        return make_llm(self.cfg, self.cfg.shorten_llm)

    @cached_property
    def tts(self) -> TTSEngine:
        return make_tts(self.cfg)

    def release_models(self) -> None:
        """Felszabadítja a betöltött modelleket, hogy a következő lépésnek legyen memóriája."""
        if "tts" in self.__dict__:
            self.__dict__.pop("tts").close()
        for name in ("llm", "shortener"):
            if name in self.__dict__:
                self.__dict__[name].unload()
