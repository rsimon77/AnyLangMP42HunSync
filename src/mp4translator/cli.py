import shlex
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from . import runlock
from .config import Config
from .pipeline import STEP_NAMES, run_pipeline

app = typer.Typer(help="Angol videók magyar AI szinkronja.", no_args_is_help=True)
console = Console()


@app.command()
def dub(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Angol nyelvű videó (max. 10 perc)")],
    output: Annotated[Optional[Path], typer.Option("-o", "--output", help="Kimeneti videó")] = None,
    config: Annotated[Path, typer.Option("-c", "--config", help="Beállítófájl")] = Path("config.yaml"),
    workdir: Annotated[Optional[Path], typer.Option("--workdir", help="Köztes fájlok mappája")] = None,
    from_step: Annotated[
        Optional[str], typer.Option("--from-step", help=f"Újrafuttatás ettől a lépéstől: {', '.join(STEP_NAMES)}")
    ] = None,
    llm: Annotated[Optional[str], typer.Option("--llm", help="pl. ollama:gemma3:12b")] = None,
    voice: Annotated[Optional[str], typer.Option("--voice", help="pl. hu-HU-NoemiNeural")] = None,
) -> None:
    """Lefuttatja a teljes szinkronfolyamatot (a már kész lépéseket átugorja)."""
    if from_step is not None and from_step not in STEP_NAMES:
        raise typer.BadParameter(f"Ismeretlen lépés: {from_step}. Lehetséges: {', '.join(STEP_NAMES)}")
    cfg = Config.load(config)
    if llm:
        cfg.llm = llm
    if voice:
        cfg.edge_voice = voice
    output = output or input.with_name(f"{input.stem}_hu.mp4")
    workdir = workdir or Path("work") / input.stem

    resume = shlex.join(["uv", "run", "mp4translator", "dub", str(input), "-o", str(output), "--workdir", str(workdir)])
    try:
        ctx = run_pipeline(input, output, cfg, workdir, from_step=from_step, console=console)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Leállítva.[/bold yellow] Az eddigi eredmények mentve vannak.")
        console.print(f"Folytatás: {resume}")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[bold red]Hiba:[/bold red] {e}")
        console.print(f"A javítás után folytatható (a kész lépések megmaradnak): {resume}")
        raise typer.Exit(1)
    console.print(f"\n[bold green]Kész:[/bold green] {output}")
    console.print(f"Riport: {ctx.workdir / 'report.md'}")


def _workdirs(workdir: Path | None) -> list[Path]:
    if workdir is not None:
        return [workdir]
    root = Path("work")
    return sorted(p.parent for p in root.glob("*/manifest.json")) if root.exists() else []


@app.command()
def stop(
    workdir: Annotated[
        Optional[Path], typer.Option("--workdir", help="Csak ezt a futást állítsa le (alapból mindet a work/ alatt)")
    ] = None,
) -> None:
    """Szelíden leállítja a futó feldolgozást; az állapot megmarad, a dub paranccsal folytatható."""
    stopped = 0
    for wd in _workdirs(workdir):
        pid = runlock.request_stop(wd)
        if pid is not None:
            console.print(f"Leállítás kérve: {wd} (PID {pid})")
            stopped += 1
    if not stopped:
        console.print("Nincs futó feldolgozás.")


@app.command()
def status(
    workdir: Annotated[Optional[Path], typer.Option("--workdir", help="Csak ennek a futásnak az állapota")] = None,
) -> None:
    """Kiírja a feldolgozások állapotát (kész lépések, megszakított lépés, szegmensek)."""
    from .manifest import Manifest
    from .steps.synthesize import is_synthesized
    from .steps.voices import describe

    dirs = _workdirs(workdir)
    if not dirs:
        console.print("Nincs feldolgozás a work/ mappában.")
    for wd in dirs:
        m = Manifest.load(wd / "manifest.json")
        pid = runlock.running_pid(wd)
        state = f"[green]fut (PID {pid})[/green]" if pid else "[dim]nem fut[/dim]"
        console.print(f"[bold]{wd}[/bold] – {Path(m.source).name} – {state}")
        for name in STEP_NAMES:
            if name in m.steps_done:
                mark = "✓"
            elif name == m.in_progress:
                mark = "…" if pid else "⏸"
            else:
                mark = "·"
            console.print(f"  {mark} {name}")
        if m.segments:
            translated = sum(1 for s in m.segments if s.text_hu)
            voiced = sum(1 for s in m.segments if is_synthesized(s, wd))
            console.print(
                f"  szegmensek: {len(m.segments)}, lefordítva: {translated}, hang kész: {voiced}, "
                f"beszélők: {len(m.speakers)}"
            )
        for name, sp in m.speakers.items():
            if sp.kind:
                console.print(f"    {describe(name, sp)}")
        if m.kept_sounds:
            console.print(f"  megtartott állathangok: {len(m.kept_sounds)}")


@app.command()
def speakers(
    workdir: Annotated[Path, typer.Option("--workdir", help="A videó munkamappája (a transcribe lépés után)")],
    thresholds: Annotated[str, typer.Option(help="Kipróbálandó küszöbértékek vesszővel")] = "0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7",
    min_seconds: Annotated[Optional[float], typer.Option(help="Legkisebb beszélő (mp); alapból a config szerint")] = None,
    preview: Annotated[Optional[float], typer.Option(help="Ehhez a küszöbhöz hangmintákat is ment meghallgatásra")] = None,
    config: Annotated[Path, typer.Option("-c", "--config")] = Path("config.yaml"),
) -> None:
    """Megmutatja, hány beszélőt találna a szétválogatás különböző küszöbértékekkel (a speaker_threshold hangolásához)."""
    import numpy as np
    import soundfile as sf

    from .audio import read_mono
    from .context import Context
    from .manifest import Manifest
    from .steps.voices import cluster_speakers, pick_reference, segment_embeddings, vocals_file

    cfg = Config.load(config)
    m = Manifest.load(workdir / "manifest.json")
    if not m.segments:
        raise typer.BadParameter("Ebben a munkamappában még nincsenek szegmensek (előbb a transcribe lépés kell).")
    ctx = Context(cfg=cfg, input_path=Path(m.source), output_path=Path("-"), workdir=workdir, manifest=m)
    min_s = cfg.min_speaker_seconds if min_seconds is None else min_seconds

    y, sr = read_mono(vocals_file(ctx))
    try:
        embedded, embs = segment_embeddings(ctx, y, sr)
    finally:
        ctx.release_models()
    durations = [m.segments[i].end - m.segments[i].start for i in embedded]

    console.print(f"{len(embedded)} szegmens vizsgálva, legkisebb beszélő: {min_s:g} mp\n")
    console.print("küszöb  beszélők  beszédidő beszélőnként (mp)")
    values = [float(t) for t in thresholds.split(",")] + ([preview] if preview is not None else [])
    for t in sorted(set(values)):
        labels = cluster_speakers(embs, durations, t, min_s)
        totals = [sum(d for d, lab in zip(durations, labels) if lab == k) for k in range(max(labels) + 1)]
        console.print(f"{t:>5.2f}  {len(totals):>8}  " + ", ".join(f"{s:.0f}" for s in totals))

        if preview is not None and abs(t - preview) < 1e-9:
            out = workdir / "speakers" / f"preview_{t:.2f}"
            out.mkdir(parents=True, exist_ok=True)
            for old in out.glob("*.wav"):
                old.unlink()
            gap = np.zeros(int(0.3 * sr), dtype=np.float32)
            for k in range(len(totals)):
                rows = [r for r, lab in enumerate(labels) if lab == k]
                c = embs[rows].mean(axis=0)
                cands = [(m.segments[embedded[r]].start, m.segments[embedded[r]].end, float(embs[r] @ c)) for r in rows]
                parts = pick_reference(cands, 12.0)
                clip = np.concatenate([np.concatenate([y[int(a * sr) : int(b * sr)], gap]) for a, b in parts])
                sf.write(str(out / f"SPK{k}.wav"), clip, sr)
            console.print(f"       hangminták: {out}")


@app.command()
def ui(
    port: Annotated[int, typer.Option(help="Port (csak a 127.0.0.1-en figyel)")] = 7860,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Böngésző megnyitása")] = True,
) -> None:
    """Helyi webes felület (angol/magyar): feltöltés, beállítások, futtatás, eredmény, beszélők."""
    from .webui.app import launch

    launch(Path.cwd(), port=port, open_browser=open_browser)


@app.command()
def steps() -> None:
    """Kiírja a lépések sorrendjét."""
    for i, name in enumerate(STEP_NAMES, 1):
        console.print(f"{i}. {name}")


if __name__ == "__main__":
    app()
