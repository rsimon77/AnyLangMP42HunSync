"""[9] Kimeneti videó: eredeti képsáv újrakódolás nélkül + magyar (és opcionálisan angol) hang, SRT, riport."""

from ..audio import run_ffmpeg
from ..context import Context
from ..text import to_srt
from ..voice_kind import KIND_NAMES

NAME = "mux"


def hz(f0: float | None) -> str:
    return f"{f0:.0f} Hz" if f0 else "–"


def write_report(ctx: Context) -> None:
    segs = ctx.manifest.segments
    counts: dict[str, int] = {}
    for s in segs:
        counts[s.action or "?"] = counts.get(s.action or "?", 0) + 1
    lines = [
        f"# Szinkron riport – {ctx.input_path.name}",
        "",
        f"Szegmensek: {len(segs)}  ",
        "Illesztés: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
        "",
        "## Beszélők",
        "",
        "| beszélő | típus | forrás | alaphang | klón alaphang | korrekció | beszéd |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, sp in ctx.manifest.speakers.items():
        lines.append(
            f"| {name} | {KIND_NAMES.get(sp.kind or '', '?')} | {sp.kind_source or '–'} | {hz(sp.f0)} | "
            f"{hz(sp.clone_f0)} | {sp.pitch_shift:+.1f} félhang | {sp.seconds:.0f} mp |"
        )
    lines += ["", "## Megtartott állathangok", ""]
    if ctx.manifest.kept_sounds:
        lines += [f"- {k.start:.1f}–{k.end:.1f} mp: {k.label} ({k.score:.2f})" for k in ctx.manifest.kept_sounds]
    else:
        lines.append("Nincs.")
    lines += [
        "",
        "## Figyelmet igénylő szegmensek",
        "",
        "| # | idő | művelet | gyorsítás | rövidítve | magyar szöveg |",
        "|---|---|---|---|---|---|",
    ]
    for s in segs:
        if s.action in ("ok", "borrow") and s.shorten_count == 0 and s.speed_factor <= 1.001:
            continue
        lines.append(
            f"| {s.id} | {s.start:.1f}–{s.end:.1f} | {s.action} | {s.speed_factor:.2f}× | {s.shorten_count}× | {s.text_hu} |"
        )
    ctx.path("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(ctx: Context) -> None:
    out = ctx.output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(ctx.input_path), "-i", str(ctx.path("dub_mix.wav")), "-map", "0:v:0", "-map", "1:a:0"]
    if ctx.cfg.keep_original_audio:
        args += ["-map", "0:a:0"]
    args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    args += ["-metadata:s:a:0", "language=hun", "-metadata:s:a:0", "title=Magyar szinkron", "-disposition:a:0", "default"]
    if ctx.cfg.keep_original_audio:
        args += ["-metadata:s:a:1", "language=eng", "-metadata:s:a:1", "title=Eredeti", "-disposition:a:1", "0"]
    if out.suffix.lower() in (".mp4", ".m4v", ".mov"):
        args += ["-movflags", "+faststart"]
    run_ffmpeg(*args, str(out))

    if ctx.cfg.write_srt:
        entries = [
            (s.start, s.speech_end, s.text_hu or "") for s in ctx.manifest.segments
        ]
        out.with_suffix(".hu.srt").write_text(to_srt(entries), encoding="utf-8")
    write_report(ctx)
