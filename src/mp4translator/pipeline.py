"""A lépések sorrendje és a folytatható futtatás."""

import gc
import signal
import time
from pathlib import Path

from . import runlock
from .audio import require_ffmpeg
from .config import Config
from .context import Context
from .manifest import Manifest
from .steps import align, extract, mix, mux, separate, sounds, synthesize, transcribe, translate, voices

STEPS = [extract, separate, transcribe, sounds, voices, translate, synthesize, align, mix, mux]
STEP_NAMES = [s.NAME for s in STEPS]


def _sigterm_as_interrupt(signum, frame):
    raise KeyboardInterrupt


def run_pipeline(
    input_path: Path,
    output_path: Path,
    cfg: Config,
    workdir: Path,
    from_step: str | None = None,
    console=None,
) -> Context:
    """Lefuttatja a hiányzó lépéseket.

    Megszakításkor (Ctrl+C, SIGTERM, `mp4translator stop`) az állapot a manifestben marad: a kész lépések
    és a megszakított lépés kész szegmensei a következő futáskor nem futnak újra.
    """
    require_ffmpeg()
    workdir.mkdir(parents=True, exist_ok=True)
    manifest_path = workdir / "manifest.json"
    if manifest_path.exists():
        manifest = Manifest.load(manifest_path)
        if manifest.source != str(input_path.resolve()):
            raise RuntimeError(f"A(z) {workdir} munkamappa egy másik videóhoz tartozik: {manifest.source}")
    else:
        manifest = Manifest(source=str(input_path.resolve()))

    if from_step is not None:
        keep = STEP_NAMES[: STEP_NAMES.index(from_step)]
        manifest.steps_done = [s for s in manifest.steps_done if s in keep]
        manifest.in_progress = None

    ctx = Context(cfg=cfg, input_path=input_path, output_path=output_path, workdir=workdir, manifest=manifest)
    if console is not None:
        ctx.console = console

    runlock.acquire(workdir)
    previous_term = signal.signal(signal.SIGTERM, _sigterm_as_interrupt)
    # Háttérben indítva a SIGINT örökölten tiltva lehet; a Ctrl+C mindig szakítsa meg a futást.
    previous_int = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        for step in STEPS:
            if step.NAME in manifest.steps_done:
                ctx.console.print(f"[dim]✓ {step.NAME} (korábbi eredmény)[/dim]")
                continue
            ctx.resuming = manifest.in_progress == step.NAME
            ctx.console.print(f"[bold cyan]▶ {step.NAME}[/bold cyan]" + (" (folytatás)" if ctx.resuming else ""))
            manifest.in_progress = step.NAME
            ctx.save()
            t0 = time.monotonic()
            step.run(ctx)
            manifest.steps_done.append(step.NAME)
            manifest.in_progress = None
            ctx.save()
            ctx.console.print(f"  kész ({time.monotonic() - t0:.0f} mp)")
            ctx.release_models()
            gc.collect()
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        try:
            ctx.release_models()
        finally:
            runlock.release(workdir)
    return ctx
