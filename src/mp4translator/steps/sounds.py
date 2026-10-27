"""[3a] Állathangok (ugatás, nyávogás stb.) keresése a beszédszünetekben.

A Demucs az állathangokat gyakran az énekhang-sávba teszi, a Whisper viszont nem írja át őket, így a magyar
sávból kimaradnának. Az AST hangesemény-felismerő (AudioSet-címkék) a beszédszünetekben megkeresi őket; a mix
lépés ezeket a részeket változatlanul visszakeveri. A beszéd alatti részeket nem vizsgáljuk: ott a magas,
rajzfilmes hangokat tévesen állathangnak nézi.
"""

import numpy as np

from ..audio import read_mono
from ..context import Context
from ..devices import pick_device
from ..manifest import KeptSound

NAME = "sounds"

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
WINDOW_S, HOP_S = 2.0, 1.0
MARGIN_S = 0.15  # ennyit hagyunk a beszéd előtt és után
MIN_GAP_S = 0.5
SILENCE_RMS = 0.005

# Az AudioSet „Animal” ágának címkéi (ezeket tartjuk meg)
ANIMAL_LABELS = {
    "Animal", "Domestic animals, pets", "Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)",
    "Cat", "Purr", "Meow", "Caterwaul", "Livestock, farm animals, working animals", "Horse", "Neigh, whinny",
    "Cattle, bovinae", "Moo", "Pig", "Oink", "Goat", "Bleat", "Sheep", "Fowl", "Chicken, rooster", "Cluck",
    "Crowing, cock-a-doodle-doo", "Turkey", "Gobble", "Duck", "Quack", "Goose", "Honk", "Wild animals",
    "Roaring cats (lions, tigers)", "Roar", "Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet",
    "Squawk", "Pigeon, dove", "Coo", "Crow", "Caw", "Owl", "Hoot", "Canidae, dogs, wolves", "Rodents, rats, mice",
    "Mouse", "Insect", "Cricket", "Frog", "Croak", "Whale vocalization",
}


def speech_gaps(spans: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """A beszéd nélküli szakaszok (a beszéd körül kis ráhagyással)."""
    gaps, cursor = [], 0.0
    for start, end in sorted(spans):
        if start - MARGIN_S - cursor >= MIN_GAP_S:
            gaps.append((cursor, start - MARGIN_S))
        cursor = max(cursor, end + MARGIN_S)
    if duration - cursor >= MIN_GAP_S:
        gaps.append((cursor, duration))
    return gaps


def windows(gaps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for a, b in gaps:
        t = a
        while True:
            out.append((t, min(t + WINDOW_S, b)))
            if t + WINDOW_S >= b:
                break
            t += HOP_S
    return out


def merge_hits(hits: list[tuple[float, float, str, float]], max_gap: float = 0.3) -> list[KeptSound]:
    """Az átfedő vagy közeli találatok összevonása; a címke a legerősebb találaté."""
    merged: list[KeptSound] = []
    for start, end, label, score in sorted(hits):
        last = merged[-1] if merged else None
        if last and start <= last.end + max_gap:
            last.end = max(last.end, end)
            if score > last.score:
                last.label, last.score = label, score
        else:
            merged.append(KeptSound(start=start, end=end, label=label, score=round(score, 3)))
    return merged


def run(ctx: Context) -> None:
    m, cfg = ctx.manifest, ctx.cfg
    m.kept_sounds = []
    vocals = ctx.path("vocals.wav")
    if not cfg.keep_animal_sounds or cfg.background != "separate" or not vocals.exists():
        ctx.console.print("  kihagyva (csak leválasztott háttérrel értelmes)")
        return

    import librosa
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    y, sr = read_mono(vocals)
    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    wins = windows(speech_gaps([(s.start, s.end) for s in m.segments], y.size / SR))
    clips = [(a, b, y[int(a * SR) : int(b * SR)]) for a, b in wins]
    clips = [c for c in clips if c[2].size >= SR // 4 and float(np.sqrt(np.mean(c[2] ** 2))) > SILENCE_RMS]
    if not clips:
        ctx.console.print("  nincs hang a beszédszünetekben")
        return

    device = pick_device(cfg.torch_device)
    fe = ASTFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).eval().to(device)
    labels = model.config.id2label
    animal_ids = [i for i, name in labels.items() if name in ANIMAL_LABELS]

    hits = []
    for k in range(0, len(clips), 16):
        chunk = clips[k : k + 16]
        inputs = fe([c[2] for c in chunk], sampling_rate=SR, return_tensors="pt").to(device)
        with torch.inference_mode():
            probs = torch.sigmoid(model(**inputs).logits).float().cpu().numpy()
        for (a, b, _), p in zip(chunk, probs):
            best = max(animal_ids, key=lambda i: p[i])
            if p[best] >= cfg.animal_threshold:
                hits.append((a, b, labels[best], float(p[best])))
    del model
    if device == "mps":
        torch.mps.empty_cache()

    m.kept_sounds = merge_hits(hits)
    ctx.console.print(f"  {len(clips)} hangos részlet a szünetekben, {len(m.kept_sounds)} állathang megtartva")
    for s in m.kept_sounds:
        ctx.console.print(f"    {s.start:6.1f}–{s.end:6.1f} mp  {s.label} ({s.score:.2f})")
