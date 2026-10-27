"""LLM háttérrendszerek a fordításhoz."""

from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Config

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMBackend(Protocol):
    def complete_json(self, system: str, user: str, schema: type[T]) -> T: ...

    def complete_text(self, system: str, user: str) -> str: ...

    def unload(self) -> None: ...


class OllamaBackend:
    def __init__(self, model: str, host: str, temperature: float = 0.3, num_ctx: int = 8192):
        self.model = model
        self.host = host.rstrip("/")
        self.options = {"temperature": temperature, "num_ctx": num_ctx}
        self.client = httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0))
        self._thinks: bool | None = None
        self.tokens, self.seconds = 0, 0.0  # generált tokenek és az erre fordított idő (méréshez)

    def thinks(self) -> bool:
        """Tud-e a modell „gondolkodni” (Ollama capabilities); egyszer kérdezzük le."""
        if self._thinks is None:
            try:
                r = self.client.post(f"{self.host}/api/show", json={"model": self.model})
            except httpx.HTTPError:
                return False
            self._thinks = r.is_success and "thinking" in r.json().get("capabilities", [])
        return self._thinks

    def _chat(self, system: str, user: str, fmt: dict | None) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": self.options,
        }
        if fmt is not None:
            body["format"] = fmt
        if self.thinks():
            body["think"] = False  # gondolkodó modell (qwen3, gemma4…): csak a választ kérjük, az sokkal gyorsabb
        try:
            r = self.client.post(f"{self.host}/api/chat", json=body)
        except httpx.ConnectError as e:
            raise LLMError(
                f"Az Ollama nem érhető el itt: {self.host}. Indítsd el: `brew services start ollama` vagy `ollama serve`."
            ) from e
        except httpx.HTTPError as e:  # pl. időtúllépés
            raise LLMError(f"Ollama-hiba: {e}") from e
        if r.status_code == 404:
            raise LLMError(f"A(z) {self.model} modell nincs letöltve. Futtasd: ollama pull {self.model}")
        if r.is_error:  # pl. 500: kifogyott a memória, vagy összeomlott a modell
            raise LLMError(f"Ollama-hiba ({r.status_code}): {r.text[:300]}")
        data = r.json()
        self.tokens += data.get("eval_count", 0)
        self.seconds += data.get("eval_duration", 0) / 1e9
        return data["message"]["content"]

    def stats(self) -> str:
        rate = f", {self.tokens / self.seconds:.1f} token/mp" if self.seconds else ""
        return f"{self.tokens} token{rate}"

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        last_error: Exception | None = None
        for _ in range(3):
            content = self._chat(system, user, schema.model_json_schema())
            try:
                return schema.model_validate_json(content)
            except ValidationError as e:
                last_error = e
        raise LLMError(f"Érvénytelen JSON válasz az LLM-től: {last_error}")

    def complete_text(self, system: str, user: str) -> str:
        return self._chat(system, user, None).strip()

    def unload(self) -> None:
        """Felszabadítja a modell memóriáját a következő lépések előtt."""
        try:
            self.client.post(f"{self.host}/api/generate", json={"model": self.model, "keep_alive": 0})
        except httpx.HTTPError:
            pass


def make_llm(cfg: Config, spec: str | None = None) -> LLMBackend:
    spec = spec or cfg.llm
    provider, _, model = spec.partition(":")
    if provider == "ollama" and model:
        return OllamaBackend(model, cfg.ollama_host, num_ctx=cfg.llm_num_ctx)
    raise ValueError(f"Ismeretlen LLM beállítás: {spec!r} (várt forma: ollama:<modell>)")
