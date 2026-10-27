"""[8] A magyar beszéd és a háttér keverése, hangerő-normalizálás."""

import numpy as np
import soundfile as sf

from ..audio import duck_envelope, loudnorm, read_mono
from ..context import Context

NAME = "mix"


FADE_S = 0.03


def kept_sounds(ctx: Context, n: int) -> np.ndarray:
    """Az énekhang-sávból megtartott állathangok (a sounds lépés találatai), rövid be- és kiúsztatással."""
    out = np.zeros(n, dtype=np.float32)
    if not ctx.manifest.kept_sounds:
        return out
    vocals, sr = read_mono(ctx.path("vocals.wav"))
    fade = int(FADE_S * sr)
    for s in ctx.manifest.kept_sounds:
        a, b = int(s.start * sr), min(int(s.end * sr), n, vocals.size)
        if b - a <= 2 * fade:
            continue
        env = np.ones(b - a, dtype=np.float32)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        out[a:b] += vocals[a:b] * env
    return out


def run(ctx: Context) -> None:
    sr = ctx.cfg.sample_rate
    dub, _ = sf.read(str(ctx.path("dub_vocals.wav")), dtype="float32")

    if ctx.cfg.background == "separate":
        bg, _ = sf.read(str(ctx.path("background.wav")), dtype="float32", always_2d=True)
        n = max(bg.shape[0], dub.size)
        bg = np.pad(bg, ((0, n - bg.shape[0]), (0, 0)))
        dub = np.pad(dub, (0, n - dub.size))
        mixed = bg + dub[:, None] + kept_sounds(ctx, n)[:, None]
    elif ctx.cfg.background == "duck":
        # Demucs nélkül az eredeti hang szól tovább, a beszéd alatt erősen lehalkítva.
        orig, _ = sf.read(str(ctx.path("audio.wav")), dtype="float32", always_2d=True)
        n = max(orig.shape[0], dub.size)
        orig = np.pad(orig, ((0, n - orig.shape[0]), (0, 0)))
        dub = np.pad(dub, (0, n - dub.size))
        speech = [(s.start, max(s.end, s.start + (s.tts_duration or 0.0))) for s in ctx.manifest.segments]
        env = duck_envelope(speech, n, sr, ctx.cfg.duck_db)
        mixed = orig * env[:, None] + dub[:, None]
    else:
        mixed = np.repeat(dub[:, None], 2, axis=1)

    raw = ctx.path("dub_mix_raw.wav")
    sf.write(str(raw), np.clip(mixed, -1.0, 1.0), sr)
    loudnorm(raw, ctx.path("dub_mix.wav"), sr)
