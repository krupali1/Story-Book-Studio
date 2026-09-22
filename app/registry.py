"""Provider catalogue + per-role selection.

Three roles can each use a different provider/model:
  brain  - the agent itself: conversation, story, prompts, planning
  critic - looks at generated images and reviews them (needs a vision model)
  image  - makes the pictures

Selection order for each role: saved in Settings UI > env var > first
configured provider > demo (mock). API keys only ever come from env vars."""
from __future__ import annotations

import os

from . import config
from .providers.base import ProviderError
from .providers.image import GeminiImage, MockImage, OpenAIImage, StabilityImage
from .providers.llm import AnthropicLLM, GeminiLLM, MockLLM, OpenAILLM


def _env(*names: str) -> str:
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            return v
    return ""


# "models" are suggestions only: the UI lets you type any model id, so new
# releases work without a code change. Check each vendor's docs for current ids.
LLM_SPECS: dict[str, dict] = {
    "anthropic": {
        "label": "Anthropic (Claude)", "keys": ["ANTHROPIC_API_KEY"],
        "default": "claude-sonnet-5",
        "models": ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"],
    },
    "openai": {
        "label": "OpenAI", "keys": ["OPENAI_API_KEY"],
        "default": "gpt-5", "models": ["gpt-5", "gpt-5-mini", "gpt-4.1"],
    },
    "gemini": {
        "label": "Google Gemini", "keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "default": "gemini-2.5-pro", "models": ["gemini-2.5-pro", "gemini-2.5-flash"],
    },
    "openai_compatible": {
        "label": "OpenAI-compatible (OpenRouter, Ollama, Groq...)", "keys": ["COMPAT_BASE_URL"],
        "default": "", "models": [],
        "note": "Set COMPAT_BASE_URL (and COMPAT_API_KEY if needed). Pick a vision-capable model to use it as critic.",
    },
    "mock": {"label": "Demo (no API, scripted)", "keys": [], "default": "demo", "models": ["demo"]},
}

IMAGE_SPECS: dict[str, dict] = {
    "openai": {
        "label": "OpenAI Images", "keys": ["OPENAI_API_KEY"], "default": "gpt-image-1",
        "models": ["gpt-image-1"], "supports_reference": True,
    },
    "gemini": {
        "label": "Google Gemini Images", "keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "default": "gemini-2.5-flash-image", "models": ["gemini-2.5-flash-image"], "supports_reference": True,
    },
    "stability": {
        "label": "Stability AI", "keys": ["STABILITY_API_KEY"], "default": "core",
        "models": ["core", "ultra"], "supports_reference": False,
    },
    "mock": {"label": "Demo (placeholder art)", "keys": [], "default": "demo", "models": ["demo"],
             "supports_reference": True},
}


def configured(specs: dict, provider: str) -> bool:
    spec = specs.get(provider)
    if not spec:
        return False
    if provider == "mock":
        return True
    return bool(_env(*spec["keys"]))


def _first_configured(specs: dict) -> str:
    for p in specs:
        if p != "mock" and configured(specs, p):
            return p
    return "mock"


def resolve_role(role: str) -> dict:
    """Return {"provider", "model", "configured", "fallback"} for a role."""
    specs = IMAGE_SPECS if role == "image" else LLM_SPECS
    saved = config.load_settings().get(role, {}) or {}
    provider = saved.get("provider") or _env(f"{role.upper()}_PROVIDER")
    model = saved.get("model") or _env(f"{role.upper()}_MODEL")

    if role == "critic" and not provider:  # default the critic to the brain's provider
        b = resolve_role("brain")
        provider, model = b["provider"], model or b["model"]

    fallback = False
    if not provider or provider not in specs:
        provider, model, fallback = _first_configured(specs), "", False
    elif not configured(specs, provider):
        provider, model, fallback = _first_configured(specs), "", True  # saved choice lost its key
    if not model:
        model = _env("COMPAT_MODEL") if provider == "openai_compatible" else specs[provider]["default"]
    return {"provider": provider, "model": model, "configured": configured(specs, provider), "fallback": fallback}


def build_llm(role: str):
    sel = resolve_role(role)
    p, m = sel["provider"], sel["model"]
    if p == "anthropic":
        return AnthropicLLM(_env("ANTHROPIC_API_KEY"), m, _env("ANTHROPIC_BASE_URL") or "https://api.anthropic.com")
    if p == "openai":
        return OpenAILLM(_env("OPENAI_API_KEY"), m, _env("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    if p == "gemini":
        return GeminiLLM(_env("GEMINI_API_KEY", "GOOGLE_API_KEY"), m)
    if p == "openai_compatible":
        if not m:
            raise ProviderError("Pick a model for the OpenAI-compatible provider in Settings (or set COMPAT_MODEL).")
        return OpenAILLM(_env("COMPAT_API_KEY"), m, _env("COMPAT_BASE_URL"), compat=True, label="Compatible endpoint")
    return MockLLM()


def build_image():
    sel = resolve_role("image")
    p, m = sel["provider"], sel["model"]
    if p == "openai":
        return OpenAIImage(_env("OPENAI_API_KEY"), m, _env("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    if p == "gemini":
        return GeminiImage(_env("GEMINI_API_KEY", "GOOGLE_API_KEY"), m)
    if p == "stability":
        return StabilityImage(_env("STABILITY_API_KEY"), m)
    return MockImage()


def describe() -> dict:
    """Everything the Settings UI needs. Never includes key values."""
    def rows(specs: dict) -> list[dict]:
        return [{"id": k, "label": v["label"], "configured": configured(specs, k), "models": v["models"],
                 "default_model": v["default"], "note": v.get("note", ""),
                 "supports_reference": v.get("supports_reference"), "env": v["keys"]}
                for k, v in specs.items()]

    roles = {r: resolve_role(r) for r in config.ROLES}
    return {"roles": roles, "llm": rows(LLM_SPECS), "image": rows(IMAGE_SPECS),
            "demo_mode": roles["brain"]["provider"] == "mock"}


def image_supports_reference() -> bool:
    sel = resolve_role("image")
    return bool(IMAGE_SPECS[sel["provider"]].get("supports_reference"))
