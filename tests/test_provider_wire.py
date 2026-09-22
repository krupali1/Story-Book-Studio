"""Checks that each adapter builds the request its vendor documents and parses
the documented response shape. These use httpx.MockTransport, so they prove our
side of the wire format, not that a live key works."""
import asyncio
import base64
import json
from io import BytesIO

import httpx
import pytest
from PIL import Image

from app import registry
from app.providers import base
from app.providers.base import Msg, ProviderError
from app.providers.image import GeminiImage, OpenAIImage, StabilityImage
from app.providers.llm import AnthropicLLM, GeminiLLM, OpenAILLM


def png(color=(200, 30, 30)) -> bytes:
    b = BytesIO()
    Image.new("RGB", (8, 8), color).save(b, "PNG")
    return b.getvalue()


def use(handler):
    base.TRANSPORT = httpx.MockTransport(handler)


def run(c):
    return asyncio.run(c)


MSGS = [Msg("user", "hello", [png()]), Msg("assistant", "hi"), Msg("user", "again")]


def test_anthropic_request_and_parse():
    seen = {}

    def h(req):
        seen.update(url=str(req.url), headers=req.headers, body=json.loads(req.content))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]})

    use(h)
    out = run(AnthropicLLM("k", "claude-sonnet-5").complete("SYS", MSGS))
    assert out == "AB"
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "k" and seen["headers"]["anthropic-version"] == "2023-06-01"
    b = seen["body"]
    assert b["model"] == "claude-sonnet-5" and b["system"] == "SYS" and b["max_tokens"] > 0
    first = b["messages"][0]["content"]
    assert first[0]["type"] == "image" and first[0]["source"]["media_type"] == "image/png"
    assert first[1] == {"type": "text", "text": "hello"}
    assert [m["role"] for m in b["messages"]] == ["user", "assistant", "user"]


def test_openai_request_and_parse_and_compat_token_param():
    seen = []

    def h(req):
        seen.append((str(req.url), req.headers, json.loads(req.content)))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    use(h)
    assert run(OpenAILLM("sk", "gpt-5").complete("SYS", MSGS)) == "ok"
    url, headers, body = seen[0]
    assert url == "https://api.openai.com/v1/chat/completions" and headers["authorization"] == "Bearer sk"
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert "max_completion_tokens" in body and "max_tokens" not in body
    part = body["messages"][1]["content"]
    assert part[0]["type"] == "text" and part[1]["image_url"]["url"].startswith("data:image/png;base64,")

    run(OpenAILLM("", "llama3", "http://localhost:11434/v1", compat=True).complete("S", [Msg("user", "x")]))
    url, headers, body = seen[1]
    assert url == "http://localhost:11434/v1/chat/completions" and "authorization" not in headers
    assert "max_tokens" in body and "max_completion_tokens" not in body


def test_gemini_llm_request_and_parse():
    seen = {}

    def h(req):
        seen.update(url=str(req.url), headers=req.headers, body=json.loads(req.content))
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "hey"}]}}]})

    use(h)
    assert run(GeminiLLM("gk", "gemini-2.5-pro").complete("SYS", MSGS)) == "hey"
    assert seen["url"].endswith("/v1beta/models/gemini-2.5-pro:generateContent")
    assert seen["headers"]["x-goog-api-key"] == "gk"
    b = seen["body"]
    assert b["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert [c["role"] for c in b["contents"]] == ["user", "model", "user"]
    assert "inlineData" in b["contents"][0]["parts"][0]


def test_gemini_llm_blocked_response_is_a_clear_error():
    use(lambda req: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(ProviderError, match="no candidates"):
        run(GeminiLLM("k", "m").complete("s", [Msg("user", "x")]))


def test_error_mapping_and_retry(monkeypatch):
    async def nosleep(_): return None
    monkeypatch.setattr(base.asyncio, "sleep", nosleep)
    calls = {"n": 0}

    def h(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"content": [{"type": "text", "text": "finally"}]})

    use(h)
    assert run(AnthropicLLM("k", "m").complete("s", [Msg("user", "x")])) == "finally" and calls["n"] == 3

    use(lambda req: httpx.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}))
    with pytest.raises(ProviderError, match="401.*bad key"):
        run(AnthropicLLM("k", "m").complete("s", [Msg("user", "x")]))


def test_normalize_merges_and_starts_with_user():
    out = base.normalize([Msg("assistant", "a"), Msg("user", "b"), Msg("user", "c")])
    assert [m.role for m in out] == ["user", "assistant", "user"] and out[2].text == "b\n\nc"


def test_openai_image_generations_vs_edits():
    seen = []
    b64png = base64.b64encode(png()).decode()

    def h(req):
        seen.append((str(req.url), req.headers.get("content-type", ""), req.content))
        return httpx.Response(200, json={"data": [{"b64_json": b64png}]})

    use(h)
    img = OpenAIImage("sk")
    assert run(img.generate("a fox", "blurry", "portrait", [])) == png()
    url, ctype, content = seen[0]
    body = json.loads(content)
    assert url.endswith("/images/generations") and ctype.startswith("application/json")
    assert body["model"] == "gpt-image-1" and body["size"] == "1024x1536" and "Avoid: blurry" in body["prompt"]
    assert "response_format" not in body

    run(img.generate("a fox", "", "square", [png(), png((1, 2, 3))]))
    url, ctype, content = seen[1]
    assert url.endswith("/images/edits") and ctype.startswith("multipart/form-data")
    assert content.count(b'name="image[]"') == 2 and b'name="prompt"' in content


def test_gemini_image_parses_inline_data_and_retries_without_ratio():
    b64png = base64.b64encode(png()).decode()
    bodies = []

    def h(req):
        body = json.loads(req.content)
        bodies.append(body)
        if "imageConfig" in body["generationConfig"]:
            return httpx.Response(400, json={"error": {"message": "unknown field imageConfig"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": "here"}, {"inlineData": {"mimeType": "image/png", "data": b64png}}]}}]})

    use(h)
    out = run(GeminiImage("gk").generate("a fox", "", "landscape", [png()]))
    assert out == png() and len(bodies) == 2
    parts = bodies[0]["contents"][0]["parts"]
    assert "inlineData" in parts[0] and "reference" in parts[-1]["text"]
    assert bodies[0]["generationConfig"]["imageConfig"]["aspectRatio"] == "4:3"
    assert bodies[0]["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]


def test_gemini_image_text_only_refusal_raises():
    use(lambda req: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "I can't draw that"}]}}]}))
    with pytest.raises(ProviderError, match="no image.*can't draw"):
        run(GeminiImage("k").generate("x", "", "square", []))


def test_stability_multipart_and_image_response():
    seen = {}

    def h(req):
        seen.update(url=str(req.url), headers=req.headers, content=req.content)
        return httpx.Response(200, content=png(), headers={"content-type": "image/png"})

    use(h)
    out = run(StabilityImage("stk", "ultra").generate("a fox", "blurry", "portrait", [png()]))
    assert out == png()
    assert seen["url"].endswith("/v2beta/stable-image/generate/ultra")
    assert seen["headers"]["authorization"] == "Bearer stk" and seen["headers"]["accept"] == "image/*"
    assert seen["headers"]["content-type"].startswith("multipart/form-data")
    assert b'name="aspect_ratio"' in seen["content"] and b"2:3" in seen["content"]
    assert b'name="negative_prompt"' in seen["content"]


def test_registry_selection_rules(monkeypatch):
    # nothing configured -> demo
    assert registry.resolve_role("brain")["provider"] == "mock"
    assert registry.describe()["demo_mode"] is True
    # a key appears -> first configured wins; critic follows brain; image picks its own
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert registry.resolve_role("brain") == {"provider": "anthropic", "model": "claude-sonnet-5",
                                             "configured": True, "fallback": False}
    assert registry.resolve_role("critic")["provider"] == "anthropic"
    assert registry.resolve_role("image")["provider"] == "openai"
    # env override, then saved settings override env
    monkeypatch.setenv("BRAIN_PROVIDER", "openai")
    assert registry.resolve_role("brain")["provider"] == "openai"
    from app import config
    config.save_settings({"brain": {"provider": "anthropic", "model": "claude-opus-5"},
                          "image": {"provider": "gemini", "model": ""}})
    assert registry.resolve_role("brain")["model"] == "claude-opus-5"
    # saved provider that lost its key falls back and says so
    r = registry.resolve_role("image")
    assert r["fallback"] is True and r["provider"] == "openai"
    assert isinstance(registry.build_llm("brain"), AnthropicLLM)
    assert isinstance(registry.build_image(), OpenAIImage)


def test_compat_provider_needs_model(monkeypatch):
    monkeypatch.setenv("COMPAT_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("BRAIN_PROVIDER", "openai_compatible")
    with pytest.raises(ProviderError, match="model"):
        registry.build_llm("brain")
    monkeypatch.setenv("COMPAT_MODEL", "llama3.2-vision")
    assert isinstance(registry.build_llm("brain"), OpenAILLM)
