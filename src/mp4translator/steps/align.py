"""[7] Időzítés-illesztés: döntés szegmensenként, majd a magyar beszédsáv összeállítása."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import read_mono, time_stretch
from ..config import Config
from ..context import Context, StepError

NAME = "align"


@dataclass
class Fit:
    action: str  # ok | borrow | speedup | shorten | forced_speedup | overflow
    speed: float = 1.0


def plan_fit(tts_dur: float, start: float, end: float, next_start: float, cfg: Config, can_shorten: bool) -> Fit:
    slot = end - start
    available = max(slot + max(next_start - end, 0.0) * cfg.gap_borrow_ratio, 1e-3)
    if tts_dur <= slot:
        return Fit("ok")
    if tts_dur <= available:
        return Fit("borrow")
    ratio = tts_dur / available
    if ratio <= cfg.soft_speedup:
        return Fit("speedup", ratio)
    if can_shorten:
        return Fit("shorten")
    if ratio <= cfg.max_speedup:
        return Fit("forced_speedup", ratio)
    return Fit("overflow", cfg.max_speedup)


def run(ctx: Context) -> None:
    m, sr = ctx.manifest, ctx.cfg.sample_rate
    track = np.zeros(int(np.ceil(m.duration * sr)), dtype=np.float32)

    for seg in m.segments:
        if not seg.tts_file:
            raise StepError(f"A(z) {seg.id}. szegmenshez nincs hangfájl – előbb a synthesize lépés kell.")
        src = ctx.workdir / seg.tts_file
        speaker = m.speakers.get(seg.speaker)
        shift = speaker.pitch_shift if speaker else 0.0
        if seg.speed_factor > 1.001 or shift:
            stretched = ctx.path(f"tts/seg_{seg.id:04d}_fit.wav")
            time_stretch(src, stretched, seg.speed_factor, shift)
            src = stretched
        y, file_sr = read_mono(src)
        if file_sr != sr:
            raise StepError(f"Váratlan mintavétel ({file_sr} Hz) a(z) {Path(src).name} fájlban.")
        a = int(round(seg.start * sr))
        b = min(a + y.size, track.size)
        track[a:b] += y[: b - a]

    sf.write(str(ctx.path("dub_vocals.wav")), np.clip(track, -1.0, 1.0), sr)
