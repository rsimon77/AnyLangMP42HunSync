"""ffmpeg- és numpy-alapú hangsegédfüggvények."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


class FFmpegError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FFmpegError(f"A(z) {tool} nem található. Telepítés: brew install ffmpeg")


def run_ffmpeg(*args: str) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg hiba ({' '.join(cmd)}):\n{proc.stderr.strip()}")


def probe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"Nem olvasható médiafájl: {path}\n{proc.stderr.strip()}")
    return json.loads(proc.stdout)


def to_wav(src: Path, dst: Path, sample_rate: int, channels: int) -> None:
    run_ffmpeg("-i", str(src), "-vn", "-ac", str(channels), "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst))


def time_stretch(src: Path, dst: Path, factor: float, semitones: float = 0.0) -> None:
    """Hangmagasság-megtartó gyorsítás (factor > 1 → rövidebb), opcionális hangmagasság-eltolással.

    Az eltolás a mintavételi ráta átírásával történik, ami a formánsokat is elviszi – gyerekhangnál ez kívánatos.
    """
    sr = sf.info(str(src)).samplerate
    if abs(semitones) < 1e-3:
        filt = f"atempo={factor:.4f}"
    else:
        k = 2 ** (semitones / 12)
        filt = f"asetrate={sr * k:.0f},aresample={sr},atempo={factor / k:.4f}"
    run_ffmpeg("-i", str(src), "-filter:a", filt, "-ar", str(sr), str(dst))


def loudnorm(src: Path, dst: Path, sample_rate: int) -> None:
    run_ffmpeg("-i", str(src), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", str(sample_rate), str(dst))


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def duration_of(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def trim_silence(y: np.ndarray, sr: int, threshold_db: float = -40.0, pad_s: float = 0.03) -> np.ndarray:
    """Levágja az elejéről és végéről a küszöb alatti csendet."""
    if y.size == 0:
        return y
    threshold = 10 ** (threshold_db / 20) * max(float(np.abs(y).max()), 1e-9)
    loud = np.flatnonzero(np.abs(y) > threshold)
    if loud.size == 0:
        return y[:0]
    pad = int(pad_s * sr)
    return y[max(loud[0] - pad, 0) : min(loud[-1] + pad + 1, y.size)]


def duck_envelope(
    intervals: list[tuple[float, float]],
    n_samples: int,
    sr: int,
    duck_db: float,
    margin_s: float = 0.1,
    ramp_s: float = 0.15,
) -> np.ndarray:
    """Erősítésgörbe: a beszédszakaszok alatt duck_db, máshol 1.0, lineáris átmenetekkel."""
    rate = 100  # a görbe felbontása (Hz)
    n = int(np.ceil(n_samples / sr * rate)) + 1
    low = np.ones(n, dtype=np.float32)
    gain = np.float32(10 ** (duck_db / 20))
    for start, end in intervals:
        a = max(int((start - margin_s) * rate), 0)
        b = min(int(np.ceil((end + margin_s) * rate)), n)
        low[a:b] = gain
    k = max(int(ramp_s * rate), 1)
    padded = np.pad(low, (k, k), mode="edge")
    smooth = np.convolve(padded, np.ones(k, dtype=np.float32) / k, mode="same")[k:-k]
    t_low = np.arange(n) / rate
    t = np.arange(n_samples) / sr
    return np.interp(t, t_low, smooth).astype(np.float32)
