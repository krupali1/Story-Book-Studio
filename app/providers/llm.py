"""Text + vision model adapters. Every adapter exposes:

    await adapter.complete(system: str, messages: list[Msg], max_tokens: int) -> str

Adding a provider = write one class with that method and register it in
app/registry.py."""
from __future__ import annotations

import json
import re

from .base import Msg, ProviderError, b64, normalize, post, raise_for, sniff_mime


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.anthropic.com"):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")

    async def complete(self, system: str, messages: list[Msg], max_tokens: int = 8000) -> str:
        msgs = []
        for m in normalize(messages):
            parts = [
                {"type": "image", "source": {"type": "base64", "media_type": sniff_mime(i), "data": b64(i)}}
                for i in m.images
            ]
            parts.append({"type": "text", "text": m.text or "(empty)"})
            msgs.append({"role": m.role, "content": parts})
        body = {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": msgs}
        r = await post(
            f"{self.base_url}/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json=body,
        )
        raise_for(r, "Anthropic")
        data = r.json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class OpenAILLM:
    """OpenAI chat completions. With compat=True it also serves any
    OpenAI-compatible endpoint (OpenRouter, Ollama, Groq, Together, vLLM...)."""

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1",
                 compat: bool = False, label: str = "OpenAI"):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self.compat, self.label = compat, label

    async def complete(self, system: str, messages: list[Msg], max_tokens: int = 8000) -> str:
        msgs: list[dict] = [{"role": "system", "content": system}]
        for m in normalize(messages):
            if m.images:
                parts: list[dict] = [{"type": "text", "text": m.text or "(empty)"}]
                for i in m.images:
                    parts.append({"type": "image_url",
                                  "image_url": {"url": f"data:{sniff_mime(i)};base64,{b64(i)}"}})
                msgs.append({"role": m.role, "content": parts})
            else:
                msgs.append({"role": m.role, "content": m.text or "(empty)"})
        body: dict = {"model": self.model, "messages": msgs}
        # Newer OpenAI models want max_completion_tokens; most compatible servers still use max_tokens.
        body["max_tokens" if self.compat else "max_completion_tokens"] = max_tokens
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        r = await post(f"{self.base_url}/chat/completions", headers=headers, json=body)
        raise_for(r, self.label)
        data = r.json()
        try:
            content = data["choices"][0]["message"].get("content")
        except (KeyError, IndexError):
            raise ProviderError(f"{self.label} returned no choices: {str(data)[:200]}")
        if isinstance(content, list):  # some servers return parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""


class GeminiLLM:
    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://generativelanguage.googleapis.com"):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")

    async def complete(self, system: str, messages: list[Msg], max_tokens: int = 8000) -> str:
        contents = []
        for m in normalize(messages):
            parts = [{"inlineData": {"mimeType": sniff_mime(i), "data": b64(i)}} for i in m.images]
            parts.append({"text": m.text or "(empty)"})
            contents.append({"role": "model" if m.role == "assistant" else "user", "parts": parts})
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        r = await post(
            f"{self.base_url}/v1beta/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key, "content-type": "application/json"},
            json=body,
        )
        raise_for(r, "Gemini")
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            raise ProviderError(f"Gemini returned no candidates: {str(data.get('promptFeedback', data))[:250]}")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            raise ProviderError(f"Gemini returned empty text (finishReason={cands[0].get('finishReason')})")
        return text


class MockLLM:
    """Scripted stand-in so the whole studio can be demoed and tested without
    API keys. It behaves like a tiny agent: it emits tool calls, reads results,
    and finishes with a summary."""

    TOOL_RESULT_PREFIX = "TOOL RESULTS"

    async def complete(self, system: str, messages: list[Msg], max_tokens: int = 8000) -> str:
        if "ART DIRECTOR REVIEW" in system:
            return ("Character consistency: n/a (demo mode)\nPalette adherence: 7/10\n"
                    "Artifacts: none visible\nText-safe area: yes\nVERDICT: pass\nFIXES: none")
        # Count tool-result rounds since the last real user message.
        k = 0
        last_user_text = ""
        for m in reversed(messages):
            if m.role == "user" and m.text.startswith(self.TOOL_RESULT_PREFIX):
                k += 1
            elif m.role == "user":
                last_user_text = m.text
                break
        tool_msgs = [m.text for m in messages if m.role == "user" and m.text.startswith(self.TOOL_RESULT_PREFIX)]
        ids = re.findall(r"img_[0-9a-f]{8}", tool_msgs[-1]) if tool_msgs else []

        def block(calls: list[dict]) -> str:
            return "```tool_calls\n" + json.dumps(calls, indent=2) + "\n```"

        idea = (last_user_text.splitlines() or ["a gentle bedtime story"])[0][:80]
        if k == 0:
            return (f"**Demo mode.** No AI provider is configured, so I'm a scripted stand-in. "
                    f"Here's the shape of a real session for: _{idea}_\n\n"
                    "I'll draft a character sheet and one key page.\n" + block([
                        {"tool": "generate_image", "args": {
                            "prompt": f"Character reference sheet, friendly fox cub, soft round shapes. Idea: {idea}",
                            "purpose": "character_sheet", "aspect": "landscape"}},
                        {"tool": "generate_image", "args": {
                            "prompt": "Fox cub under a giant moon, calm sky area for text", "purpose": "page",
                            "aspect": "portrait"}}]))
        if k == 1 and ids:
            return "Two drafts are ready. Let me pull the palette, review the first one and build a moodboard.\n" + block([
                {"tool": "extract_palette", "args": {"image_id": ids[0], "n": 5}},
                {"tool": "critique_image", "args": {"image_id": ids[0], "criteria": "palette and friendliness"}},
                {"tool": "make_moodboard", "args": {"image_ids": ids[:2], "title": "Demo direction"}}])
        if k == 2:
            hexes = re.findall(r"#[0-9a-fA-F]{6}", tool_msgs[-1]) or ["#1d2350", "#ffc83d"]
            imgs = re.findall(r"img_[0-9a-f]{8}", "\n".join(tool_msgs))[:1]
            return "Saving this direction to the library and logging feedback.\n" + block([
                {"tool": "save_asset", "args": {"type": "palette", "name": "Demo moon palette",
                                                  "description": "Palette from the demo run",
                                                  "tags": ["demo", "night"], "data": {"colors": hexes},
                                                  "image_ids": imgs}},
                {"tool": "log_feedback", "args": {"verdict": "approved", "tags": ["love palette"],
                                                    "note": "demo feedback", "image_id": imgs[0] if imgs else None}}])
        return ("Done. In a real session I'd now wait for you to approve the direction before generating "
                "more pages. Pick a real provider in **Settings** to start designing for real.")
