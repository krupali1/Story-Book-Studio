"""Real-API smoke test for the provider adapters in app/providers/.

Unlike tests/ (which mock every request), this script fires one minimal, cheap
request per configured provider at the *real* vendor endpoint. It exists
because app/providers/*.py wire formats are otherwise only ever verified
against mocks (see CLAUDE.md "Known gaps"). Run it after adding real keys to
.env, before trusting a provider in production use.

Usage:
    source .venv/bin/activate
    python scripts/smoke_test_providers.py            # test every configured provider
    python scripts/smoke_test_providers.py anthropic openai   # test a subset

A provider with no key set is skipped, not failed. Never prints key values.
Exits non-zero if any configured provider's request failed.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: only fills vars not already set in the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


load_dotenv(ROOT / ".env")

from app.providers.base import Msg, ProviderError  # noqa: E402
from app.providers.image import GeminiImage, OpenAIImage, StabilityImage  # noqa: E402
from app.providers.llm import AnthropicLLM, GeminiLLM, OpenAILLM  # noqa: E402


def env(*names: str) -> str:
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            return v
    return ""


def masked(v: str) -> str:
    return f"set (…{v[-4:]})" if len(v) >= 4 else "set" if v else "not set"


async def check_llm(name: str, build) -> tuple[bool, str]:
    llm = build()
    try:
        reply = await llm.complete(
            system="You are a smoke test. Follow instructions exactly, no extra words.",
            messages=[Msg(role="user", text="Reply with exactly the single word: pong")],
            max_tokens=16,
        )
        ok = "pong" in reply.lower()
        return ok, f"reply={reply.strip()[:80]!r}" if ok else f"unexpected reply={reply.strip()[:120]!r}"
    except ProviderError as e:
        return False, str(e)
    except Exception as e:  # network errors, etc.
        return False, f"{type(e).__name__}: {e}"


async def check_image(name: str, build) -> tuple[bool, str]:
    img = build()
    try:
        data = await img.generate(
            prompt="a single small red circle centered on a plain white background, minimal, flat",
            negative="text, watermark",
            aspect="square",
            references=[],
        )
        from io import BytesIO

        from PIL import Image

        im = Image.open(BytesIO(data))
        im.verify()
        out = ROOT / "scripts" / f"_smoke_{name}.png"
        out.write_bytes(data)
        return True, f"{len(data)} bytes, format={im.format}, size={im.size} -> saved to {out.relative_to(ROOT)}"
    except ProviderError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


LLM_CHECKS = {
    "anthropic": (["ANTHROPIC_API_KEY"],
                  lambda: AnthropicLLM(env("ANTHROPIC_API_KEY"), env("ANTHROPIC_MODEL") or "claude-sonnet-5",
                                       env("ANTHROPIC_BASE_URL") or "https://api.anthropic.com")),
    "openai": (["OPENAI_API_KEY"],
               lambda: OpenAILLM(env("OPENAI_API_KEY"), env("OPENAI_LLM_MODEL") or "gpt-5",
                                  env("OPENAI_BASE_URL") or "https://api.openai.com/v1")),
    "gemini": (["GEMINI_API_KEY", "GOOGLE_API_KEY"],
               lambda: GeminiLLM(env("GEMINI_API_KEY", "GOOGLE_API_KEY"), env("GEMINI_LLM_MODEL") or "gemini-2.5-pro")),
    "openai_compatible": (["COMPAT_BASE_URL"],
                           lambda: OpenAILLM(env("COMPAT_API_KEY"), env("COMPAT_MODEL"), env("COMPAT_BASE_URL"),
                                              compat=True, label="Compatible endpoint")),
}

IMAGE_CHECKS = {
    "openai_image": (["OPENAI_API_KEY"],
                      lambda: OpenAIImage(env("OPENAI_API_KEY"), env("OPENAI_IMAGE_MODEL") or "gpt-image-1")),
    "gemini_image": (["GEMINI_API_KEY", "GOOGLE_API_KEY"],
                      lambda: GeminiImage(env("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                                          env("GEMINI_IMAGE_MODEL") or "gemini-2.5-flash-image")),
    "stability": (["STABILITY_API_KEY"],
                  lambda: StabilityImage(env("STABILITY_API_KEY"), env("STABILITY_MODEL") or "core")),
}


async def main() -> int:
    only = set(sys.argv[1:]) or None
    rows: list[tuple[str, str, bool | None, str]] = []

    for name, (keys, build) in LLM_CHECKS.items():
        if only and name not in only:
            continue
        if name == "openai_compatible" and not env(*keys):
            rows.append((name, "llm", None, f"skipped — {keys[0]} not set"))
            continue
        if not env(*keys):
            rows.append((name, "llm", None, f"skipped — {' / '.join(keys)} not set"))
            continue
        ok, msg = await check_llm(name, build)
        rows.append((name, "llm", ok, msg))

    for name, (keys, build) in IMAGE_CHECKS.items():
        if only and name not in only:
            continue
        if not env(*keys):
            rows.append((name, "image", None, f"skipped — {' / '.join(keys)} not set"))
            continue
        ok, msg = await check_image(name, build)
        rows.append((name, "image", ok, msg))

    print(f"{'provider':<20} {'kind':<7} {'result':<8} detail")
    print("-" * 90)
    any_fail = False
    for name, kind, ok, msg in rows:
        result = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        if ok is False:
            any_fail = True
        print(f"{name:<20} {kind:<7} {result:<8} {msg}")

    print()
    print("Key status:", ", ".join(
        f"{n}={masked(env(*k))}" for n, k in {
            "ANTHROPIC_API_KEY": ["ANTHROPIC_API_KEY"], "OPENAI_API_KEY": ["OPENAI_API_KEY"],
            "GEMINI_API_KEY": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "STABILITY_API_KEY": ["STABILITY_API_KEY"],
        }.items()
    ))
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
