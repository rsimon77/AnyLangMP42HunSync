"""Ollama-modellek lekérdezése és letöltése (a választólistákhoz és a letöltés gombhoz)."""

import json
from collections.abc import Iterator

import httpx


def installed_models(host: str) -> list[str] | None:
    """A letöltött modellek nevei (pl. gemma3:12b); None, ha az Ollama nem fut."""
    try:
        r = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=3)
        r.raise_for_status()
    except httpx.HTTPError:
        return None
    return sorted(m["name"] for m in r.json().get("models", []))


def is_installed(tag: str, installed: list[str]) -> bool:
    """A „gemma3” név a „gemma3:latest” címkét jelenti."""
    return tag in installed or (":" not in tag and f"{tag}:latest" in installed)


def pull(host: str, tag: str) -> Iterator[tuple[str, float | None]]:
    """Letölti a modellt; (állapot, haladás 0–1 vagy None) párokat ad vissza menet közben."""
    body = {"model": tag, "stream": True}
    with httpx.stream("POST", f"{host.rstrip('/')}/api/pull", json=body, timeout=httpx.Timeout(None, connect=5)) as r:
        if r.is_error:
            r.read()
            raise RuntimeError(f"Ollama-hiba ({r.status_code}): {r.text[:300]}")
        for line in r.iter_lines():
            if not line.strip():
                continue
            msg = json.loads(line)
            if "error" in msg:
                raise RuntimeError(msg["error"])
            total, done = msg.get("total"), msg.get("completed")
            yield msg.get("status", ""), (done / total if total and done is not None else None)
