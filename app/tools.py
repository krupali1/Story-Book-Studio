"""Tools the agent can call. Each tool takes (ctx, args: dict) and returns a
ToolResult. Tools never raise to the agent: failures come back as ok=False with
a plain-language message the agent can act on."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from . import images, library, registry
from .providers.base import Msg, ProviderError

ASPECTS = ("square", "portrait", "landscape")


@dataclass
class Ctx:
    session_id: str
    images_left: int


@dataclass
class ToolResult:
    ok: bool
    text: str                      # what the agent reads
    summary: str = ""              # short line for the UI chip
    images: list[str] = field(default_factory=list)  # image ids to show in the chat


TOOLS: dict[str, dict] = {}


def tool(name: str, sig: str, desc: str):
    def deco(fn):
        TOOLS[name] = {"fn": fn, "sig": sig, "desc": desc}
        return fn
    return deco


def manual() -> str:
    return "\n".join(f"- `{n}` {t['sig']}\n  {t['desc']}" for n, t in TOOLS.items())


def _ids(v) -> list[str]:
    if not v:
        return []
    if isinstance(v, str):
        v = [v]
    return [str(x) for x in v]


def _missing(ids: list[str]) -> list[str]:
    return [i for i in ids if not images.exists(i)]


@tool("search_library", '{"query": str, "type": "character|style|palette|book" (optional)}',
      "Search saved characters, styles, palettes and books. Empty query lists the most recent.")
async def search_library(ctx: Ctx, a: dict) -> ToolResult:
    rows = library.search(a.get("query", ""), a.get("type") or None)
    if not rows:
        return ToolResult(True, "No matches.", "0 matches")
    text = "\n".join(f"- {r['type']}/{r['name']}: {r['description'][:140]} tags={r['tags']} images={r['image_ids'][:2]}"
                     for r in rows)
    return ToolResult(True, text, f"{len(rows)} matches")


@tool("read_asset", '{"type": str, "name": str}',
      "Read the full saved data of one library item (e.g. a character bible or a style prompt).")
async def read_asset(ctx: Ctx, a: dict) -> ToolResult:
    asset = library.get_asset(a.get("type", ""), a.get("name", ""))
    if not asset:
        return ToolResult(False, f"Nothing saved as {a.get('type')}/{a.get('name')}.", "not found")
    return ToolResult(True, json.dumps(asset, ensure_ascii=False)[:7000], f"read {asset['name']}", asset["image_ids"][:3])


@tool("generate_image",
      '{"prompt": str, "purpose": "character_sheet|page|cover|other", "aspect": "square|portrait|landscape", '
      '"style": "saved style name" (optional), "reference_ids": ["img_..."] (optional), "negative": str (optional)}',
      "Create one image with the image provider. Pass the saved style name and the approved character sheet id "
      "in reference_ids to keep the book consistent. Costs money; there is a per-turn image budget.")
async def generate_image(ctx: Ctx, a: dict) -> ToolResult:
    prompt = (a.get("prompt") or "").strip()
    if not prompt:
        return ToolResult(False, "prompt is required", "no prompt")
    if ctx.images_left <= 0:
        return ToolResult(False, "Image budget for this turn is used up. Show what you have and ask the user "
                                 "to say 'continue' for more.", "budget reached")
    aspect = a.get("aspect") if a.get("aspect") in ASPECTS else "square"
    refs = _ids(a.get("reference_ids"))
    if bad := _missing(refs):
        return ToolResult(False, f"Unknown reference image ids: {bad}", "bad reference")

    style = library.get_asset("style", a["style"]) if a.get("style") else None
    if a.get("style") and not style:
        return ToolResult(False, f"No saved style named '{a['style']}'. Save it first with save_asset "
                                 f"(type=style) or omit the style.", "style not found")
    style_data = (style or {}).get("data", {})
    full = "\n\n".join(p for p in (style_data.get("style_prompt"), prompt,
                                   "No text, letters, captions or watermarks anywhere in the image.") if p)
    negative = ", ".join(p for p in (style_data.get("negative_prompt"), a.get("negative"),
                                     "text, watermark, extra fingers, deformed hands, blurry") if p)

    provider = registry.build_image()
    sel = registry.resolve_role("image")
    note = ""
    use_refs = refs
    if refs and not provider.supports_reference:
        use_refs, note = [], (f" NOTE: {sel['provider']} cannot take reference images, so they were ignored. "
                              f"Describe the character fully in the prompt instead.")
    ctx.images_left -= 1
    try:
        data = await provider.generate(full, negative, aspect, [images.for_vision(i, 1536) for i in use_refs])
        image_id = await asyncio.to_thread(images.save, data, {
            "prompt": full, "negative": negative, "purpose": a.get("purpose", "other"), "aspect": aspect,
            "style": a.get("style"), "reference_ids": use_refs, "provider": sel["provider"],
            "model": sel["model"], "session_id": ctx.session_id})
    except ProviderError as e:
        return ToolResult(False, f"Image generation failed: {e}", "generation failed")
    except Exception as e:  # corrupt image bytes etc.
        return ToolResult(False, f"Image generation failed: {e!r}", "generation failed")
    return ToolResult(True, f"Generated {image_id} ({aspect}, {sel['provider']}/{sel['model']}, "
                            f"references: {use_refs or 'none'}).{note}", f"{a.get('purpose', 'image')} → {image_id}",
                      [image_id])


@tool("critique_image",
      '{"image_id": "img_...", "criteria": str (optional), "reference_ids": ["img_..."] (optional), '
      '"style": str (optional)}',
      "Have the critic model review an image: character consistency vs the reference sheet, palette and style "
      "adherence, anatomy/artifacts, stray text, age-appropriateness, space for text. Returns a verdict and fixes.")
async def critique_image(ctx: Ctx, a: dict) -> ToolResult:
    image_id, refs = a.get("image_id", ""), _ids(a.get("reference_ids"))
    if bad := _missing([image_id] + refs):
        return ToolResult(False, f"Unknown image ids: {bad}", "bad image id")
    m = images.meta(image_id)
    style = library.get_asset("style", a["style"]) if a.get("style") else None
    system = (
        "ART DIRECTOR REVIEW. You are an exacting art director for illustrated children's picture books. "
        "Review the LAST image. Any earlier images are reference sheets showing the approved character/style. "
        "Reply in under 140 words with these lines: Character consistency (vs reference, or n/a); "
        "Palette and style adherence; Artifacts (hands, extra limbs, warped faces, stray text or letters); "
        "Age-appropriate tone; Room for page text; VERDICT: pass|regenerate; FIXES: concrete prompt changes."
    )
    text = (f"Intended prompt:\n{m.get('prompt', '(unknown)')}\n\n"
            + (f"Style notes: {json.dumps(style['data'])[:600]}\n\n" if style else "")
            + (f"Extra criteria: {a['criteria']}\n\n" if a.get("criteria") else "")
            + f"{len(refs)} reference image(s) come first, then the image to review.")
    try:
        critic = registry.build_llm("critic")
        msg = Msg("user", text, [images.for_vision(i) for i in refs] + [images.for_vision(image_id)])
        out = await critic.complete(system, [msg], 1200)
    except ProviderError as e:
        return ToolResult(False, f"Critique failed: {e}", "critique failed")
    verdict = "pass" if "VERDICT: pass" in out else "regenerate" if "VERDICT: regenerate" in out else "reviewed"
    return ToolResult(True, out.strip(), f"{image_id}: {verdict}")


@tool("extract_palette", '{"image_id": "img_...", "n": int (2-12, default 6)}',
      "Pull the dominant colours (hex) from an image, e.g. an approved page, to define or check a palette.")
async def extract_palette(ctx: Ctx, a: dict) -> ToolResult:
    if bad := _missing([a.get("image_id", "")]):
        return ToolResult(False, f"Unknown image id: {bad}", "bad image id")
    cols = await asyncio.to_thread(images.extract_palette, images.read(a["image_id"]), a.get("n", 6))
    return ToolResult(True, "Palette: " + ", ".join(cols), " ".join(cols))


@tool("make_moodboard", '{"image_ids": ["img_..."] (1-9), "palette": ["#rrggbb"] (optional), "title": str}',
      "Composite images and palette swatches into one board so the user can compare directions at a glance.")
async def make_moodboard(ctx: Ctx, a: dict) -> ToolResult:
    ids = _ids(a.get("image_ids"))[:9]
    if not ids:
        return ToolResult(False, "image_ids is required", "no images")
    if bad := _missing(ids):
        return ToolResult(False, f"Unknown image ids: {bad}", "bad image id")
    raw = [images.read(i) for i in ids]
    board = await asyncio.to_thread(images.moodboard, raw, _ids(a.get("palette")), a.get("title", "Moodboard"))
    new_id = await asyncio.to_thread(images.save, board, {"purpose": "moodboard", "session_id": ctx.session_id,
                                                          "prompt": a.get("title", "Moodboard")})
    return ToolResult(True, f"Moodboard created: {new_id}", f"moodboard → {new_id}", [new_id])


@tool("save_asset",
      '{"type": "character|style|palette|book", "name": str, "description": str, "tags": [str], '
      '"data": object, "image_ids": ["img_..."]}',
      "Save to the library. Conventions for data: character {bible, reference_image_id}; "
      "style {style_prompt, negative_prompt, palette, fonts, model, settings}; palette {colors, mood}; "
      "book {brief, style, characters, pages:[{n, text, image_prompt, image_id}], status, retro}. "
      "Saving the same name again updates it (version bumps).")
async def save_asset(ctx: Ctx, a: dict) -> ToolResult:
    t, name = a.get("type", ""), (a.get("name") or "").strip()
    if t not in library.TYPES or not name:
        return ToolResult(False, f"type must be one of {library.TYPES} and name is required", "invalid")
    data = a.get("data")
    if data is not None and not isinstance(data, dict):
        return ToolResult(False, "data must be a JSON object", "invalid data")
    ids = _ids(a.get("image_ids"))
    if bad := _missing(ids):
        return ToolResult(False, f"Unknown image ids: {bad}", "bad image id")
    asset = library.save_asset(t, name, a.get("description", ""), a.get("tags"), data, ids)
    return ToolResult(True, f"Saved {t} '{asset['name']}' (v{asset['version']}).", f"saved {t}: {asset['name']}", ids[:1])


@tool("log_feedback",
      '{"verdict": "approved|rejected|note", "tags": [str], "note": str, "image_id": "img_..." (optional), '
      '"book": str (optional)}',
      "Record the user's reaction with short tags (e.g. 'too dark', 'off-model', 'love palette'). Rejections with "
      "reasons are the most valuable signal. Do this whenever the user reacts to something.")
async def log_feedback(ctx: Ctx, a: dict) -> ToolResult:
    verdict = a.get("verdict") if a.get("verdict") in ("approved", "rejected", "note") else "note"
    entry = {"verdict": verdict, "tags": _ids(a.get("tags")), "note": a.get("note", ""), "book": a.get("book"),
             "session_id": ctx.session_id, "image_id": a.get("image_id")}
    if a.get("image_id") and images.valid_id(a["image_id"]):
        entry["prompt"] = images.meta(a["image_id"]).get("prompt", "")[:300]
    library.log_feedback(entry)
    return ToolResult(True, "Feedback logged.", f"{verdict}: {', '.join(entry['tags']) or 'note'}")


@tool("update_taste", '{"markdown": str}',
      "Rewrite the taste profile (max ~6000 chars). Keep it a short, distilled list of durable lessons: styles, "
      "palettes and character traits the user loves; what they reject and why. Merge with the existing profile, "
      "drop stale items, don't just append.")
async def update_taste(ctx: Ctx, a: dict) -> ToolResult:
    try:
        library.set_taste(a.get("markdown", ""))
    except ValueError as e:
        return ToolResult(False, str(e), "not saved")
    return ToolResult(True, "Taste profile updated (previous version kept in taste_history).", "taste updated")


async def run(ctx: Ctx, call: dict) -> ToolResult:
    name = call.get("tool")
    spec = TOOLS.get(name)
    if not spec:
        return ToolResult(False, f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}", "unknown tool")
    args = call.get("args") or {}
    if not isinstance(args, dict):
        return ToolResult(False, "args must be a JSON object", "bad args")
    try:
        return await spec["fn"](ctx, args)
    except Exception as e:  # last-resort guard so one bad call can't kill the turn
        return ToolResult(False, f"Tool crashed: {e!r}", "tool error")
