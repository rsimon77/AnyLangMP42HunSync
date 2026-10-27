import json

import httpx
import pytest

from mp4translator.config import Config
from mp4translator.llm import LLMError, OllamaBackend
from mp4translator.manifest import Segment
from mp4translator.models import DEMUCS, LLMS, TTS_ENGINES, WHISPERS, describe_all, ollama_tag
from mp4translator.ollama import is_installed
from mp4translator.steps.extract import media_duration
from mp4translator.context import StepError


def test_catalog_defaults_match_config():
    cfg = Config()
    for catalog, value in [(LLMS, cfg.llm), (WHISPERS, cfg.whisper_model), (DEMUCS, cfg.demucs_model), (TTS_ENGINES, cfg.tts)]:
        assert [m.id for m in catalog if m.recommended] == [value]


def test_catalog_values_are_valid_config():
    for m in TTS_ENGINES:
        Config(tts=m.id)
    assert all(m.id.startswith("ollama:") for m in LLMS)
    assert "translategemma:12b" in describe_all("hu")


def test_ollama_tag_and_installed():
    assert ollama_tag("ollama:gemma3:12b") == "gemma3:12b"
    assert is_installed("gemma3:12b", ["gemma3:12b"])
    assert is_installed("llama3", ["llama3:latest"])
    assert not is_installed("gemma3:4b", ["gemma3:12b"])


def _backend(handler) -> OllamaBackend:
    b = OllamaBackend("m", "http://ollama")
    b.client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def test_thinking_model_gets_think_false():
    sent = []

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "thinking"]})
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "szia"}, "eval_count": 10, "eval_duration": 2e9})

    b = _backend(handler)
    assert b.complete_text("s", "u") == "szia"
    assert sent[0]["think"] is False
    assert b.stats() == "10 token, 5.0 token/mp"


def test_plain_model_gets_no_think_field():
    sent = []

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion"]})
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    _backend(handler).complete_text("s", "u")
    assert "think" not in sent[0]


@pytest.mark.parametrize("status", [500, 503])
def test_server_error_is_llm_error(status):
    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        return httpx.Response(status, text="out of memory")

    with pytest.raises(LLMError, match="out of memory"):
        _backend(handler).complete_text("s", "u")


def test_timeout_is_llm_error():
    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError):
        _backend(handler).complete_text("s", "u")


def test_media_duration_falls_back_to_streams():
    assert media_duration({"format": {"duration": "12.5"}}) == 12.5
    assert media_duration({"format": {"duration": "N/A"}, "streams": [{"duration": "N/A"}, {"duration": "7"}]}) == 7.0
    with pytest.raises(StepError):
        media_duration({"format": {}, "streams": [{}]})


def test_speech_end_accounts_for_speedup():
    seg = Segment(id=0, start=1.0, end=2.0, text_en="x", tts_duration=1.5, speed_factor=1.25)
    assert seg.speech_end == pytest.approx(2.2)
    assert Segment(id=0, start=1.0, end=2.0, text_en="x").speech_end == 2.0


def test_answer_schema_requires_every_segment():
    from mp4translator.steps.translate import answer_schema, answers

    schema = answer_schema([3, 7])
    assert schema.model_json_schema()["required"] == ["s3", "s7"]
    assert answers(schema.model_validate_json('{"s3": " Szia! ", "s7": ""}')) == {3: "Szia!"}
    with pytest.raises(ValueError):
        schema.model_validate_json('{"s3": "Szia!"}')
