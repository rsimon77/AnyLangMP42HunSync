"""Ajánlott modellek a választólistákhoz (parancssor és webes felület).

A lista csak ajánlás: bármely más Ollama-modell, mlx-community Whisper-repó vagy Demucs-modell is megadható.
A méretek a letöltés méretei; a „mem” becsült memóriaigény futás közben (16 GB-os gépen fontos).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    id: str
    label: str
    size_gb: float
    note_en: str
    note_hu: str
    recommended: bool = False

    def note(self, lang: str) -> str:
        return self.note_hu if lang == "hu" else self.note_en

    def title(self, lang: str = "en") -> str:
        star = "★ " if self.recommended else ""
        return f"{star}{self.label} – {self.size_gb:g} GB – {self.note(lang)}"


# Fordító LLM-ek (Ollama). A gondolkodó modelleknél (qwen3, gemma4) az llm.py kikapcsolja a gondolkodást.
LLMS = [
    ModelChoice("ollama:gemma3:12b", "Gemma 3 12B", 8.1, "default, best tested Hungarian", "alapértelmezett, a legjobban kipróbált magyar", True),
    ModelChoice("ollama:gemma3:4b", "Gemma 3 4B", 3.3, "fast, weaker Hungarian", "gyors, gyengébb magyar"),
    ModelChoice("ollama:translategemma:12b", "TranslateGemma 12B", 8.1, "translation-tuned Gemma 3", "fordításra hangolt Gemma 3"),
    ModelChoice("ollama:translategemma:4b", "TranslateGemma 4B", 3.3, "translation-tuned, fast", "fordításra hangolt, gyors"),
    ModelChoice("ollama:gemma4:12b", "Gemma 4 12B", 7.7, "newer Gemma, experimental", "újabb Gemma, kísérleti"),
    ModelChoice("ollama:gemma4:e4b", "Gemma 4 E4B", 6.6, "newer small Gemma, experimental", "újabb kis Gemma, kísérleti"),
    ModelChoice("ollama:qwen3:8b", "Qwen3 8B", 5.2, "experimental, weaker Hungarian", "kísérleti, gyengébb magyar"),
]

# Whisper (mlx-whisper, Hugging Face repók)
WHISPERS = [
    ModelChoice("mlx-community/whisper-large-v3-turbo", "large-v3-turbo", 1.6, "default: fast and accurate", "alapértelmezett: gyors és pontos", True),
    ModelChoice("mlx-community/whisper-large-v3-turbo-4bit", "large-v3-turbo 4-bit", 0.46, "smaller, slightly less accurate", "kisebb, kicsit pontatlanabb"),
    ModelChoice("mlx-community/whisper-large-v3-mlx", "large-v3", 3.1, "most accurate, slower", "a legpontosabb, lassabb"),
    ModelChoice("mlx-community/distil-whisper-large-v3", "distil-large-v3", 1.5, "English only, fast", "csak angol, gyors"),
    ModelChoice("mlx-community/whisper-small-mlx", "small", 0.48, "small and fast, less accurate", "kicsi és gyors, pontatlanabb"),
]

# Demucs (zene/effekt és beszéd szétválasztása)
DEMUCS = [
    ModelChoice("htdemucs", "htdemucs", 0.08, "default", "alapértelmezett", True),
    ModelChoice("htdemucs_ft", "htdemucs_ft", 0.32, "cleaner voice, about 4x slower", "tisztább beszéd, kb. 4x lassabb"),
    ModelChoice("mdx_extra", "mdx_extra", 0.4, "alternative separation model", "másik szétválasztó modell"),
]

# Hangmotorok (a config `tts` mezője)
TTS_ENGINES = [
    ModelChoice("xtts", "XTTS-v2", 1.9, "clones voices, non-commercial license", "klónoz, nem kereskedelmi licenc", True),
    ModelChoice("edge", "Microsoft Edge", 0.0, "online, no cloning", "online, klónozás nélkül"),
]


def ollama_tag(spec: str) -> str:
    """"ollama:gemma3:12b" → "gemma3:12b"."""
    return spec.split(":", 1)[1] if spec.startswith("ollama:") else spec


def describe_all(lang: str = "en") -> str:
    lines = []
    for title, items in (("LLM", LLMS), ("Whisper", WHISPERS), ("Demucs", DEMUCS), ("TTS", TTS_ENGINES)):
        lines.append(f"{title}:")
        lines += [f"  {'★' if m.recommended else ' '} {m.id} ({m.size_gb:g} GB) – {m.note(lang)}" for m in items]
    return "\n".join(lines)
