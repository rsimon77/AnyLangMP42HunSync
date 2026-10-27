"""Hangtípus (férfi / női / gyerek) az alaphang alapján, és a klónozott hang hangmagasság-korrekciója."""

import math
from pathlib import Path

import numpy as np

from .config import Config

KIND_NAMES = {"male": "férfi", "female": "női", "child": "gyerek"}
KIND_PROMPT = {"male": "adult man", "female": "adult woman", "child": "child"}

F0_SR = 16000


def median_f0(y: np.ndarray, sr: int, max_seconds: float = 60.0) -> float | None:
    """A zöngés részek alaphangjának mediánja (Hz); None, ha nincs zöngés hang."""
    import librosa

    if sr != F0_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=F0_SR)
    y = y[: int(max_seconds * F0_SR)]
    if y.size < F0_SR // 4:
        return None
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=700, sr=F0_SR)
    f = f0[voiced & np.isfinite(f0)]
    return round(float(np.median(f)), 1) if f.size >= 10 else None


def median_f0_of_files(paths: list[Path]) -> float | None:
    import librosa

    from .audio import read_mono

    parts = []
    for p in paths:
        y, sr = read_mono(p)
        parts.append(y if sr == F0_SR else librosa.resample(y, orig_sr=sr, target_sr=F0_SR))
    return median_f0(np.concatenate(parts), F0_SR) if parts else None


def classify_kind(f0: float | None, cfg: Config) -> str | None:
    if f0 is None:
        return None
    if f0 < cfg.male_max_f0:
        return "male"
    if f0 >= cfg.child_min_f0:
        return "child"
    return "female"


def pitch_correction(ref_f0: float | None, clone_f0: float | None, kind: str | None, cfg: Config) -> float:
    """Hány félhanggal kell feljebb (negatív: lejjebb) vinni a klónozott hangot, hogy az eredetihez illjen."""
    if ref_f0 is None or clone_f0 is None or cfg.pitch_correction == "off":
        return 0.0
    if cfg.pitch_correction == "child" and kind != "child":
        return 0.0
    st = 12 * math.log2(ref_f0 / clone_f0)
    if abs(st) < cfg.min_pitch_shift:
        return 0.0
    return round(max(min(st, cfg.max_pitch_shift), -cfg.max_pitch_shift), 2)
