"""Cserélhető beszédszintézis-motorok."""

import asyncio
import gc
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from .audio import read_mono, to_wav, trim_silence
from .config import Config
from .devices import pick_device
from .text import count_hu_syllables


class TTSError(RuntimeError):
    pass


class TTSEngine(Protocol):
    # True, ha a motor maga tud gyorsabban beszélni (jobb minőség, mint az utólagos time-stretch)
    supports_speed: bool
    # True, ha a motor a beszélő hangmintájából klónoz
    clones_voice: bool
    # Az utolsó synthesize() hívás próbálkozásainak száma (méréshez)
    last_attempts: int

    def prepare_speaker(self, speaker: str, ref_audio: Path | None, kind: str | None = None) -> None:
        """Felkészül a beszélőre: klónozó motor a hangmintából, a többi a hangtípus (male/female/child) alapján."""
        ...

    def set_rate(self, speaker: str, syllables_per_second: float | None) -> None:
        """A beszélő klónozott hangjának mért tempója (a kimondott hossz ellenőrzéséhez)."""
        ...

    def synthesize(self, text: str, speaker: str, out: Path, speed: float = 1.0) -> float:
        """Kimondja a szöveget, mono WAV-ba menti (csend nélkül), visszaadja a hosszát mp-ben."""
        ...

    def close(self) -> None: ...


class SpeakerEmbedder(Protocol):
    def embed(self, audio: Path) -> np.ndarray:
        """Beszélő-azonosító vektor (normált) egy hangrészletből – a beszélők szétválogatásához."""
        ...


def write_atomic(out: Path, y: np.ndarray, sr: int) -> None:
    """Ideiglenes névre ír, majd átnevez: megszakításkor nem marad félig írt WAV a végleges néven."""
    tmp = out.with_name(out.stem + ".part.wav")
    sf.write(str(tmp), y, sr)
    tmp.replace(out)


class EdgeTTSEngine:
    supports_speed = True
    clones_voice = False
    last_attempts = 1

    def __init__(self, voice: str, sample_rate: int, voices: dict[str, str] | None = None, child_pitch: str = "+0Hz"):
        self.voice = voice
        self.sample_rate = sample_rate
        self.voices = voices or {}
        self.child_pitch = child_pitch
        self.speakers: dict[str, tuple[str, str]] = {}  # beszélő → (hang, hangmagasság)

    def prepare_speaker(self, speaker: str, ref_audio: Path | None, kind: str | None = None) -> None:
        voice = self.voices.get(kind, self.voice) if kind else self.voice
        self.speakers[speaker] = (voice, self.child_pitch if kind == "child" else "+0Hz")

    def set_rate(self, speaker: str, syllables_per_second: float | None) -> None:
        pass

    def synthesize(self, text: str, speaker: str, out: Path, speed: float = 1.0) -> float:
        import edge_tts

        rate = f"{round((speed - 1) * 100):+d}%"
        voice, pitch = self.speakers.get(speaker, (self.voice, "+0Hz"))
        with tempfile.TemporaryDirectory() as tmp:
            mp3 = Path(tmp) / "tts.mp3"
            wav = Path(tmp) / "tts.wav"
            for attempt in range(3):
                try:
                    asyncio.run(edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(mp3)))
                    break
                except Exception:  # hálózati hiba: rövid várakozás után újra
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
            to_wav(mp3, wav, self.sample_rate, channels=1)
            y, sr = read_mono(wav)
        y = trim_silence(y, sr)
        write_atomic(out, y, sr)
        return y.size / sr

    def close(self) -> None:
        pass


def plausible_duration(duration: float, text: str, syllables_per_second: float) -> float:
    """0 = pont a várt hossz; nagyobb érték = gyanúsabb (XTTS-nél gyakori a hallucináció vagy a csonkolás)."""
    expected = max(count_hu_syllables(text) / syllables_per_second, 0.4)
    if duration <= 0:
        return math.inf
    return abs(math.log(duration / expected))


class XTTSEngine:
    supports_speed = True
    clones_voice = True
    MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    MODEL_SR = 24000

    def __init__(self, cfg: Config):
        if not cfg.xtts_accept_license:
            raise TTSError(
                "Az XTTS-v2 a Coqui Public Model License alatt érhető el (csak nem kereskedelmi célra). "
                "Ha elfogadod, állítsd be a config.yaml-ban: xtts_accept_license: true "
                "(vagy használd a tts: edge motort)."
            )
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS

        self.cfg = cfg
        self.sample_rate = cfg.sample_rate
        self.device = pick_device(cfg.torch_device)
        api = TTS(self.MODEL, progress_bar=False)
        try:
            api.to(self.device)
        except Exception:
            self.device = "cpu"
            api.to("cpu")
        self._api = api
        self.model = api.synthesizer.tts_model
        self.latents: dict[str, tuple] = {}
        self.rates: dict[str, float] = {}
        self.last_attempts = 0

    def embed(self, audio: Path) -> np.ndarray:
        """Beszélő-azonosító vektor (512 dim., normált) egy hangrészletből."""
        _, emb = self.model.get_conditioning_latents(audio_path=[str(audio)], load_sr=22050)
        v = emb.detach().float().cpu().numpy().reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)

    def prepare_speaker(self, speaker: str, ref_audio: Path | None, kind: str | None = None) -> None:
        if ref_audio is None:
            raise TTSError(f"Az XTTS-hez hangminta kell a(z) {speaker} beszélőhöz.")
        self.latents[speaker] = self.model.get_conditioning_latents(
            audio_path=[str(ref_audio)], gpt_cond_len=12, max_ref_length=30, load_sr=22050
        )

    def set_rate(self, speaker: str, syllables_per_second: float | None) -> None:
        if syllables_per_second:
            self.rates[speaker] = syllables_per_second
        else:
            self.rates.pop(speaker, None)

    def _generate(self, text: str, speaker: str, speed: float) -> np.ndarray:
        import soxr
        import torch

        gpt_latent, spk_emb = self.latents[speaker]
        with torch.inference_mode():
            result = self.model.inference(
                text,
                "hu",
                gpt_latent,
                spk_emb,
                temperature=self.cfg.xtts_temperature,
                speed=speed,
                enable_text_splitting=len(text) > 200,
            )
        wav = np.asarray(result["wav"].cpu() if hasattr(result["wav"], "cpu") else result["wav"], dtype=np.float32)
        y = soxr.resample(wav.reshape(-1), self.MODEL_SR, self.sample_rate)
        return trim_silence(y.astype(np.float32), self.sample_rate)

    def synthesize(self, text: str, speaker: str, out: Path, speed: float = 1.0) -> float:
        if speaker not in self.latents:
            raise TTSError(f"Nincs hangminta a(z) {speaker} beszélőhöz – előbb a voices lépés kell.")
        best, best_score = None, math.inf
        # A klónozott hang mért tempója; enélkül a lassabb hangok hossza gyanúsnak tűnik → fölösleges újrapróbálás
        rate = self.rates.get(speaker, self.cfg.syllables_per_second)
        for attempt in range(1, max(self.cfg.xtts_retries, 1) + 1):
            self.last_attempts = attempt
            y = self._generate(text, speaker, speed)
            score = plausible_duration(y.size / self.sample_rate * speed, text, rate)
            if best is None or score < best_score:
                best, best_score = y, score
            if score < math.log(1.6):  # a várt hossz 0,63–1,6-szorosa elfogadható
                break
        write_atomic(out, best, self.sample_rate)
        return best.size / self.sample_rate

    def close(self) -> None:
        import torch

        self.latents.clear()
        self.rates.clear()
        del self.model, self._api
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()


def make_tts(cfg: Config) -> TTSEngine:
    if cfg.tts == "xtts":
        return XTTSEngine(cfg)
    if cfg.tts == "edge":
        return EdgeTTSEngine(cfg.edge_voice, cfg.sample_rate, cfg.edge_voices, cfg.edge_child_pitch)
    raise ValueError(f"Ismeretlen TTS motor: {cfg.tts!r}")
