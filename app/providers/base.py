"""Shared HTTP plumbing for provider adapters. Adapters talk to each vendor's
REST API directly through httpx, so there are no vendor SDK version conflicts."""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field

import httpx


class ProviderError(Exception):
    """Raised for any provider failure. The message is shown to the user."""


# Tests inject an httpx.MockTransport here.
TRANSPORT: httpx.AsyncBaseTransport | None = None


@dataclass
class Msg:
    role: str  # "user" | "assistant"
    text: str
    images: list[bytes] = field(default_factory=list)


def sniff_mime(b: bytes) -> str:
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def normalize(messages: list[Msg]) -> list[Msg]:
    """Merge consecutive same-role messages and make sure we start with a user
    turn. Several vendors reject anything else."""
    out: list[Msg] = []
    for m in messages:
        if out and out[-1].role == m.role:
            prev = out[-1]
            out[-1] = Msg(m.role, f"{prev.text}\n\n{m.text}".strip(), prev.images + m.images)
        else:
            out.append(Msg(m.role, m.text, list(m.images)))
    if out and out[0].role != "user":
        out.insert(0, Msg("user", "(conversation start)"))
    return out


def error_message(r: httpx.Response) -> str:
    try:
        j = r.json()
    except Exception:
        return r.text[:300]
    if isinstance(j, dict):
        e = j.get("error", j)
        if isinstance(e, dict):
            return str(e.get("message") or e)[:400]
        return str(e)[:400]
    return str(j)[:300]


def raise_for(r: httpx.Response, who: str) -> None:
    if r.status_code >= 400:
        raise ProviderError(f"{who} returned {r.status_code}: {error_message(r)}")


async def post(url: str, *, headers=None, json=None, data=None, files=None,
               timeout: float = 240, retries: int = 3) -> httpx.Response:
    """POST with light retry on network errors, 429 and 5xx."""
    last: Exception | None = None
    async with httpx.AsyncClient(transport=TRANSPORT, timeout=timeout) as c:
        for attempt in range(retries):
            try:
                r = await c.post(url, headers=headers, json=json, data=data, files=files)
            except httpx.HTTPError as e:
                last = ProviderError(f"Network error calling {httpx.URL(url).host}: {e!r}")
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            return r
    raise last or ProviderError("Request failed")
