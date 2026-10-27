"""[2] Beszéd és háttér (zene, zajok) szétválasztása Demucs-szal."""

import gc

from ..audio import to_wav
from ..context import Context
from ..devices import pick_device

NAME = "separate"


def run(ctx: Context) -> None:
    if ctx.cfg.background != "separate":
        ctx.console.print(f"  kihagyva (background: {ctx.cfg.background})")
        return

    import random

    import demucs.api
    import torch

    # A Demucs véletlen időeltolással dolgozik; rögzített maggal minden futás ugyanazt a beszédsávot adja,
    # így az átírás (és a mérések) is megismételhetők.
    random.seed(0)
    torch.manual_seed(0)
    device = pick_device(ctx.cfg.torch_device)
    ctx.console.print(f"  Demucs modell: {ctx.cfg.demucs_model} ({device})")
    separator = demucs.api.Separator(model=ctx.cfg.demucs_model, device=device, progress=True)
    try:
        _, stems = separator.separate_audio_file(str(ctx.path("audio.wav")))
    except RuntimeError as e:
        if device == "cpu":
            raise
        ctx.console.print(f"  [yellow]{device} hiba ({e}), újrapróbálás CPU-n[/yellow]")
        separator.update_parameter(device="cpu")
        _, stems = separator.separate_audio_file(str(ctx.path("audio.wav")))

    vocals = stems.pop("vocals")
    background = torch.stack(list(stems.values())).sum(dim=0)
    sr = separator.samplerate
    demucs.api.save_audio(vocals, str(ctx.path("vocals_raw.wav")), samplerate=sr)
    demucs.api.save_audio(background, str(ctx.path("background_raw.wav")), samplerate=sr)
    del separator, stems, vocals, background
    gc.collect()

    # Egységes formátum a további lépésekhez; a Whisper a tisztított beszédet kapja.
    to_wav(ctx.path("vocals_raw.wav"), ctx.path("vocals.wav"), ctx.cfg.sample_rate, channels=1)
    to_wav(ctx.path("background_raw.wav"), ctx.path("background.wav"), ctx.cfg.sample_rate, channels=2)
    to_wav(ctx.path("vocals_raw.wav"), ctx.path("asr_16k.wav"), 16000, channels=1)
    ctx.path("vocals_raw.wav").unlink()
    ctx.path("background_raw.wav").unlink()
