"""[3b] Beszélők szétválogatása és beszélőnkénti referencia-hangminta a hangklónozáshoz."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import read_mono
from ..context import Context
from ..manifest import Speaker
from ..text import count_hu_syllables
from ..voice_kind import KIND_NAMES, classify_kind, median_f0, median_f0_of_files, pitch_correction

NAME = "voices"

MIN_EMBED_S = 0.8  # ennél rövidebb részletből megbízhatatlan a beszélővektor


def cluster_speakers(embs: np.ndarray, durations: list[float], threshold: float, min_seconds: float) -> list[int]:
    """Normált beszélővektorok csoportosítása; a túl kevés beszédet tartalmazó csoportok a legközelebbihez olvadnak.

    A címkék a beszédidő szerint csökkenő sorrendben 0, 1, 2, …
    """
    n = len(embs)
    if n == 0:
        return []
    if n == 1:
        return [0]
    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average"
    ).fit_predict(embs)

    dur = np.asarray(durations, dtype=float)
    while True:
        ids = np.unique(labels)
        totals = {c: dur[labels == c].sum() for c in ids}
        small = [c for c in ids if totals[c] < min_seconds]
        if not small or len(ids) == 1:
            break
        c = min(small, key=totals.get)
        centroids = {k: embs[labels == k].mean(axis=0) for k in ids if k != c}
        own = embs[labels == c].mean(axis=0)
        target = max(centroids, key=lambda k: float(own @ centroids[k]) / (np.linalg.norm(centroids[k]) + 1e-9))
        labels[labels == c] = target

    order = sorted(np.unique(labels), key=lambda c: -dur[labels == c].sum())
    remap = {c: i for i, c in enumerate(order)}
    return [remap[c] for c in labels]


def nearest_in_time(index: int, labelled: dict[int, int]) -> int:
    """A legközelebbi (időben) címkézett szegmens címkéje; egyenlőségnél az előző."""
    return labelled[min(labelled, key=lambda j: (abs(j - index), j > index))]


def pick_reference(candidates: list[tuple[float, float, float]], target_s: float) -> list[tuple[float, float]]:
    """(start, end, hasonlóság) jelöltekből a legjellemzőbb, hosszabb részletek, összesen kb. target_s mp."""
    ranked = sorted(candidates, key=lambda c: c[2] * min(c[1] - c[0], 6.0), reverse=True)
    chosen, total = [], 0.0
    for start, end, _ in ranked:
        if total >= target_s:
            break
        chosen.append((start, end))
        total += end - start
    return sorted(chosen)


CALIBRATION_SENTENCES = [
    "Ma reggel korán keltem, megittam egy kávét, aztán elindultam a boltba.",
    "Szerintem ez egy nagyon jó ötlet, próbáljuk ki együtt még ma délután!",
]


def clamp_rate(syllables: int, seconds: float, low: float = 4.0, high: float = 9.0) -> float | None:
    """Szótag/mp, észszerű határok közé szorítva; None, ha nincs mérhető hang."""
    if seconds <= 0 or syllables <= 0:
        return None
    return round(min(max(syllables / seconds, low), high), 2)


def calibration_files(out_dir: Path, speaker: str) -> list[Path]:
    return [out_dir / f"{speaker}_cal{k}.wav" for k in range(len(CALIBRATION_SENTENCES))]


def measure_rate(ctx: Context, speaker: str, out_dir: Path) -> float | None:
    """Megméri a klónozott hang tempóját néhány próbamondaton, hogy a fordítás pontosabb szótagkeretet kapjon.

    A próbamondatok az out_dir-ben maradnak: ezekből mérjük a klón hangmagasságát, és meg is hallgathatók.
    """
    syllables, seconds = 0, 0.0
    for sentence, out in zip(CALIBRATION_SENTENCES, calibration_files(out_dir, speaker)):
        seconds += ctx.tts.synthesize(sentence, speaker, out)
        syllables += count_hu_syllables(sentence)
    return clamp_rate(syllables, seconds)


def speaker_kind(name: str, f0: float | None, ctx: Context) -> tuple[str | None, str]:
    """A beszélő hangtípusa: a config.yaml speaker_kinds felülírja az automatikus felismerést."""
    if name in ctx.cfg.speaker_kinds:
        return ctx.cfg.speaker_kinds[name], "config"
    return classify_kind(f0, ctx.cfg), "auto"


def describe(name: str, sp: Speaker) -> str:
    kind = KIND_NAMES.get(sp.kind or "", "ismeretlen")
    f0 = f", {sp.f0:.0f} Hz" if sp.f0 else ""
    shift = f", hangmagasság {sp.pitch_shift:+.1f} félhang" if sp.pitch_shift else ""
    return f"{name}: {kind} ({sp.kind_source}{f0}){shift}"


def vocals_file(ctx: Context) -> Path:
    p = ctx.path("vocals.wav")
    return p if p.exists() else ctx.path("audio.wav")


def segment_embeddings(ctx: Context, y: np.ndarray, sr: int) -> tuple[list[int], np.ndarray]:
    """Beszélővektor minden elég hosszú szegmenshez; gyorsítótárazva, mert a hangolás többször is kéri."""
    m = ctx.manifest
    embedded = [i for i, s in enumerate(m.segments) if s.end - s.start >= MIN_EMBED_S]
    if not embedded:
        embedded = list(range(len(m.segments)))
    spans = np.array([[m.segments[i].start, m.segments[i].end] for i in embedded])

    cache = ctx.path("speakers/embeddings.npz")
    if cache.exists():
        data = np.load(cache)
        if data["spans"].shape == spans.shape and np.allclose(data["spans"], spans):
            return embedded, data["embs"]

    embs = []
    with tempfile.TemporaryDirectory() as tmp:
        for n, i in enumerate(embedded, 1):
            p = Path(tmp) / f"{i}.wav"
            sf.write(str(p), y[int(m.segments[i].start * sr) : int(m.segments[i].end * sr)], sr)
            embs.append(ctx.tts.embed(p))
            if n % 25 == 0:
                ctx.console.print(f"  beszélővektorok: {n}/{len(embedded)}")
    embs_arr = np.stack(embs)
    np.savez(cache, spans=spans, embs=embs_arr)
    return embedded, embs_arr


def run(ctx: Context) -> None:
    m, cfg = ctx.manifest, ctx.cfg
    tts = ctx.tts
    m.speakers = {}
    for s in m.segments:
        s.tts_text = None  # új hangminták → minden szegmens hangját újra kell generálni
    y, sr = read_mono(vocals_file(ctx))

    def clip(start: float, end: float) -> np.ndarray:
        return y[int(start * sr) : int(end * sr)]

    if not tts.clones_voice:
        for s in m.segments:
            s.speaker = "SPK0"
        f0 = median_f0(np.concatenate([clip(s.start, s.end) for s in m.segments]), sr) if m.segments else None
        speaker = Speaker(ref_audio=None, seconds=sum(s.end - s.start for s in m.segments), f0=f0)
        speaker.kind, speaker.kind_source = speaker_kind("SPK0", f0, ctx)
        tts.prepare_speaker("SPK0", None, speaker.kind)
        m.speakers["SPK0"] = speaker
        ctx.console.print(f"  a TTS motor nem klónoz, egyetlen hang lesz – {describe('SPK0', speaker)}")
        return

    embedded, embs_arr = segment_embeddings(ctx, y, sr)
    for old in (ctx.workdir / "speakers").glob("SPK*.wav"):
        old.unlink()

    if cfg.diarize:
        labels = cluster_speakers(
            embs_arr,
            [m.segments[i].end - m.segments[i].start for i in embedded],
            cfg.speaker_threshold,
            cfg.min_speaker_seconds,
        )
    else:
        labels = [0] * len(embedded)
    labelled = dict(zip(embedded, labels))
    for i, s in enumerate(m.segments):
        s.speaker = f"SPK{labelled[i] if i in labelled else nearest_in_time(i, labelled)}"

    gap = np.zeros(int(0.15 * sr), dtype=np.float32)
    for label in sorted(set(labels)):
        rows = [k for k, lab in enumerate(labels) if lab == label]
        centroid = embs_arr[rows].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        candidates = [
            (m.segments[embedded[k]].start, m.segments[embedded[k]].end, float(embs_arr[k] @ centroid)) for k in rows
        ]
        parts = pick_reference(candidates, cfg.ref_audio_seconds)
        ref = np.concatenate([np.concatenate([clip(a, b), gap]) for a, b in parts])
        name = f"SPK{label}"
        ref_path = ctx.path(f"speakers/{name}.wav")
        sf.write(str(ref_path), ref, sr)
        seconds = sum(s.end - s.start for s in m.segments if s.speaker == name)
        speaker = Speaker(
            ref_audio=str(ref_path.relative_to(ctx.workdir)), seconds=round(seconds, 1), f0=median_f0(ref, sr)
        )
        speaker.kind, speaker.kind_source = speaker_kind(name, speaker.f0, ctx)
        tts.prepare_speaker(name, ref_path, speaker.kind)
        speaker.syllables_per_second = measure_rate(ctx, name, ref_path.parent)
        tts.set_rate(name, speaker.syllables_per_second)
        speaker.clone_f0 = median_f0_of_files(calibration_files(ref_path.parent, name))
        speaker.pitch_shift = pitch_correction(speaker.f0, speaker.clone_f0, speaker.kind, cfg)
        m.speakers[name] = speaker
        rate = f"{speaker.syllables_per_second:.1f} szótag/mp" if speaker.syllables_per_second else "nem mérhető"
        ctx.console.print(
            f"  {describe(name, speaker)}; {seconds:.0f} mp beszéd, {ref.size / sr:.1f} mp hangminta, tempó {rate}"
        )
