"""Image generation adapters. Every adapter exposes:

    supports_reference: bool
    await adapter.generate(prompt, negative, aspect, references) -> bytes

`aspect` is one of "square" | "portrait" | "landscape". `references` are raw
image bytes (approved character sheets etc.) for providers that accept them."""
from __future__ import annotations

import base64
import hashlib
import random
from io import BytesIO

import httpx

from . import base
from .base import ProviderError, b64, post, raise_for, sniff_mime

ASPECTS = ("square", "portrait", "landscape")


def _with_negative(prompt: str, negative: str) -> str:
    return f"{prompt}\n\nAvoid: {negative}" if negative else prompt


class OpenAIImage:
    supports_reference = True
    SIZES = {"square": "1024x1024", "portrait": "1024x1536", "landscape": "1536x1024"}

    def __init__(self, api_key: str, model: str = "gpt-image-1", base_url: str = "https://api.openai.com/v1"):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")

    async def generate(self, prompt: str, negative: str, aspect: str, references: list[bytes]) -> bytes:
        text = _with_negative(prompt, negative)
        size = self.SIZES.get(aspect, "1024x1024")
        auth = {"authorization": f"Bearer {self.api_key}"}
        if references:
            files = [("image[]", (f"ref{i}.png", ref, sniff_mime(ref))) for i, ref in enumerate(references[:4])]
            r = await post(f"{self.base_url}/images/edits", headers=auth, files=files,
                           data={"model": self.model, "prompt": text, "size": size, "n": "1"})
        else:
            r = await post(f"{self.base_url}/images/generations", headers={**auth, "content-type": "application/json"},
                           json={"model": self.model, "prompt": text, "size": size, "n": 1})
        raise_for(r, "OpenAI Images")
        item = (r.json().get("data") or [{}])[0]
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"])
        if item.get("url"):
            async with httpx.AsyncClient(transport=base.TRANSPORT, timeout=120) as c:
                got = await c.get(item["url"])
                return got.content
        raise ProviderError("OpenAI Images returned no image data")


class GeminiImage:
    supports_reference = True
    RATIOS = {"square": "1:1", "portrait": "3:4", "landscape": "4:3"}

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-image",
                 base_url: str = "https://generativelanguage.googleapis.com"):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")

    async def generate(self, prompt: str, negative: str, aspect: str, references: list[bytes]) -> bytes:
        text = _with_negative(prompt, negative)
        if references:
            text = ("Use the attached reference image(s) as the source of truth for the character's "
                    "appearance and the art style. " + text)
        parts = [{"inlineData": {"mimeType": sniff_mime(i), "data": b64(i)}} for i in references[:4]]
        parts.append({"text": text})
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}

        def body(with_ratio: bool) -> dict:
            cfg: dict = {"responseModalities": ["TEXT", "IMAGE"]}
            if with_ratio:
                cfg["imageConfig"] = {"aspectRatio": self.RATIOS.get(aspect, "1:1")}
            return {"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg}

        r = await post(url, headers=headers, json=body(True))
        if r.status_code == 400:  # some model versions reject imageConfig; retry without it
            r = await post(url, headers=headers, json=body(False))
        raise_for(r, "Gemini Images")
        data = r.json()
        for cand in data.get("candidates") or []:
            for p in (cand.get("content") or {}).get("parts") or []:
                blob = p.get("inlineData") or p.get("inline_data")
                if blob and blob.get("data"):
                    return base64.b64decode(blob["data"])
        said = " ".join(p.get("text", "") for c in data.get("candidates") or []
                        for p in (c.get("content") or {}).get("parts") or [])[:250]
        raise ProviderError(f"Gemini returned no image. {said or str(data.get('promptFeedback', ''))[:250]}")


class StabilityImage:
    supports_reference = False
    RATIOS = {"square": "1:1", "portrait": "2:3", "landscape": "3:2"}

    def __init__(self, api_key: str, model: str = "core", base_url: str = "https://api.stability.ai"):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")

    async def generate(self, prompt: str, negative: str, aspect: str, references: list[bytes]) -> bytes:
        endpoint = self.model if self.model in ("core", "ultra") else "core"
        data = {"prompt": prompt, "aspect_ratio": self.RATIOS.get(aspect, "1:1"), "output_format": "png"}
        if negative:
            data["negative_prompt"] = negative
        # Stability expects multipart even without files; the empty file part forces that encoding.
        r = await post(f"{self.base_url}/v2beta/stable-image/generate/{endpoint}",
                       headers={"authorization": f"Bearer {self.api_key}", "accept": "image/*"},
                       data=data, files={"none": ("none", b"")})
        raise_for(r, "Stability")
        if not r.headers.get("content-type", "").startswith("image/"):
            raise ProviderError(f"Stability returned non-image response: {r.text[:200]}")
        return r.content


class MockImage:
    """Draws a deterministic placeholder so the pipeline is testable offline."""
    supports_reference = True
    SIZES = {"square": (1024, 1024), "portrait": (768, 1024), "landscape": (1024, 768)}

    async def generate(self, prompt: str, negative: str, aspect: str, references: list[bytes]) -> bytes:
        from PIL import Image, ImageDraw

        w, h = self.SIZES.get(aspect, (1024, 1024))
        rnd = random.Random(hashlib.sha256(prompt.encode()).digest())

        def col() -> tuple[int, int, int]:
            return tuple(rnd.randint(70, 235) for _ in range(3))  # type: ignore[return-value]

        top, bottom = col(), col()
        img = Image.new("RGB", (w, h))
        d = ImageDraw.Draw(img)
        for y in range(h):
            t = y / h
            d.line([(0, y), (w, y)], fill=tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3)))
        for _ in range(7):
            r = rnd.randint(50, 220)
            x, y = rnd.randint(0, w), rnd.randint(0, h)
            d.ellipse([x - r, y - r, x + r, y + r], fill=col())
        d.rectangle([0, 0, w, 64], fill=(29, 35, 80))
        d.text((18, 24), "MOCK IMAGE (demo mode) - " + prompt[:70].replace("\n", " "), fill=(255, 200, 61))
        buf = BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()
