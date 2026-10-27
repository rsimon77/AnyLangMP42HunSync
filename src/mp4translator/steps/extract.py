"""[1] Bemenet ellenőrzése és a hang kinyerése."""

from ..audio import probe, to_wav
from ..context import Context, StepError

NAME = "extract"


def run(ctx: Context) -> None:
    info = probe(ctx.input_path)
    streams = info.get("streams", [])
    if not any(s.get("codec_type") == "video" for s in streams):
        raise StepError("A bemeneti fájlban nincs videósáv.")
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise StepError("A bemeneti fájlban nincs hangsáv.")

    duration = float(info["format"]["duration"])
    if duration > ctx.cfg.max_duration_s:
        raise StepError(
            f"A videó {duration / 60:.1f} perces, a megengedett maximum {ctx.cfg.max_duration_s / 60:.0f} perc."
        )
    ctx.manifest.duration = duration

    to_wav(ctx.input_path, ctx.path("audio.wav"), ctx.cfg.sample_rate, channels=2)
    to_wav(ctx.input_path, ctx.path("audio_16k.wav"), 16000, channels=1)
