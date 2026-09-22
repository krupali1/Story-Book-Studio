"""The agent loop: call the brain model, parse tool calls, run them, feed the
results back, repeat until the model hands control back to the user."""
from __future__ import annotations

import json
import re
import time
from typing import AsyncIterator

from . import config, images, prompts, registry, sessions, tools
from .providers.base import Msg, ProviderError

BLOCK_RE = re.compile(r"```tool_calls\s*(.*?)```", re.S)
MAX_CALLS_PER_STEP = 4
HISTORY_LIMIT = 60
VISION_USER_TURNS = 2  # only attachments from the latest N user messages carry pixels (keeps token cost sane)


def parse_tool_calls(raw: str) -> tuple[str, list[dict], str | None]:
    """Return (text_for_user, calls, error). Text has the tool block removed."""
    blocks = BLOCK_RE.findall(raw)
    clean = BLOCK_RE.sub("", raw).strip()
    if not blocks:
        # A lone unterminated fence is a truncation, tell the model.
        if "```tool_calls" in raw:
            return clean.split("```tool_calls")[0].strip(), [], "Your tool_calls block was cut off or not closed. Resend it."
        return clean, [], None
    calls: list[dict] = []
    for b in blocks:
        try:
            parsed = json.loads(b.strip())
        except json.JSONDecodeError as e:
            return clean, [], f"tool_calls was not valid JSON ({e.msg} at char {e.pos}). Resend a valid JSON array."
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list) or not all(isinstance(c, dict) for c in parsed):
            return clean, [], "tool_calls must be a JSON array of {\"tool\":..., \"args\":{...}} objects."
        calls.extend(parsed)
    return clean, calls, None


def _format_results(calls: list[dict]) -> str:
    lines = ["TOOL RESULTS"]
    for i, c in enumerate(calls, 1):
        lines.append(f"[{i}] {c['tool']} → {'OK' if c['ok'] else 'FAILED'}\n{c['result']}")
    lines.append("\nContinue. If everything you planned is done, reply to the user without a tool_calls block.")
    return "\n\n".join(lines)


def to_llm_messages(sess: dict) -> list[Msg]:
    msgs = sess["messages"][-HISTORY_LIMIT:]
    # Count user turns, not raw messages: a single tool-heavy turn adds many messages, and the
    # image the user just attached must stay visible to the brain for that whole turn.
    seeing = {i for i, m in enumerate(msgs) if m["kind"] == "user"}
    seeing = set(sorted(seeing)[-VISION_USER_TURNS:])
    out: list[Msg] = []
    for idx, m in enumerate(msgs):
        recent = idx in seeing
        if m["kind"] == "user":
            ids = [i for i in m.get("image_ids", []) if images.exists(i)]
            text = m["text"] + (f"\n\n[Attached images: {', '.join(ids)}]" if ids else "")
            out.append(Msg("user", text, [images.for_vision(i) for i in ids] if recent else []))
        elif m["kind"] == "assistant":
            out.append(Msg("assistant", m["raw"]))
        elif m["kind"] == "tools":
            out.append(Msg("user", _format_results(m["calls"])))
    while out and out[0].role != "user":
        out.pop(0)
    return out


async def run_turn(session_id: str, text: str, image_ids: list[str], explore: bool = False) -> AsyncIterator[dict]:
    sess = sessions.load(session_id)
    if not sess:
        yield {"type": "error", "message": "Session not found."}
        return
    image_ids = [i for i in image_ids if images.exists(i)]
    sess["messages"].append({"kind": "user", "text": text, "image_ids": image_ids,
                             "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if sess["title"] == "New book" and text.strip():
        sess["title"] = text.strip().splitlines()[0][:48]
        yield {"type": "title", "title": sess["title"]}
    sessions.save(sess)

    ctx = tools.Ctx(session_id=session_id, images_left=config.MAX_IMAGES_PER_TURN)
    try:
        brain = registry.build_llm("brain")
        system = prompts.build_system(explore, config.MAX_IMAGES_PER_TURN)
    except ProviderError as e:
        yield {"type": "error", "message": str(e)}
        return

    finished = False
    for _ in range(config.MAX_AGENT_STEPS):
        try:
            raw = await brain.complete(system, to_llm_messages(sess))
        except ProviderError as e:
            yield {"type": "error", "message": str(e)}
            sessions.save(sess)
            return
        clean, calls, err = parse_tool_calls(raw)
        sess["messages"].append({"kind": "assistant", "text": clean, "raw": raw})
        if clean:
            yield {"type": "text", "text": clean}

        if err:  # let the model repair its own formatting
            sess["messages"].append({"kind": "tools", "calls": [
                {"tool": "(format)", "args": {}, "ok": False, "result": err, "summary": "format error", "images": []}]})
            sessions.save(sess)
            continue
        if not calls:
            finished = True
            break

        results = []
        for call in calls[:MAX_CALLS_PER_STEP]:
            name = str(call.get("tool"))
            yield {"type": "tool_call", "tool": name, "args": call.get("args") or {}}
            res = await tools.run(ctx, call)
            rec = {"tool": name, "args": call.get("args") or {}, "ok": res.ok, "result": res.text,
                   "summary": res.summary, "images": res.images}
            results.append(rec)
            yield {"type": "tool_result", "tool": name, "ok": res.ok, "summary": res.summary, "images": res.images}
        if len(calls) > MAX_CALLS_PER_STEP:
            results.append({"tool": "(limit)", "args": {}, "ok": False, "summary": "too many calls", "images": [],
                            "result": f"Only the first {MAX_CALLS_PER_STEP} calls ran. Re-issue the rest if needed."})
        sess["messages"].append({"kind": "tools", "calls": results})
        sessions.save(sess)

    if not finished:
        note = "I've reached the step limit for one turn. Say **continue** and I'll pick up from here."
        sess["messages"].append({"kind": "assistant", "text": note, "raw": note})
        yield {"type": "text", "text": note}
    sessions.save(sess)
    yield {"type": "done"}
