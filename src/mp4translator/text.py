import re

HU_VOWELS = set("aáeéiíoóöőuúüűAÁEÉIÍOÓÖŐUÚÜŰ")


def count_hu_syllables(text: str) -> int:
    """Magyar szótagszám: minden magánhangzó egy szótag; a számjegyeket nagyjából becsüli."""
    vowels = sum(1 for ch in text if ch in HU_VOWELS)
    digits = sum(len(d) * 2 for d in re.findall(r"\d+", text))
    return vowels + digits


def format_srt_time(seconds: float) -> str:
    ms = max(int(round(seconds * 1000)), 0)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(entries: list[tuple[float, float, str]]) -> str:
    blocks = []
    for i, (start, end, text) in enumerate(entries, 1):
        blocks.append(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text.strip()}\n")
    return "\n".join(blocks)
