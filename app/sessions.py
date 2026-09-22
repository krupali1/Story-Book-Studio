"""Chat sessions (one per book project), stored as JSON files."""
from __future__ import annotations

import json
import re
import time
import uuid

from . import config

ID_RE = re.compile(r"^[a-f0-9]{12}$")


def _p(sid: str):
    if not ID_RE.match(sid or ""):
        raise ValueError("invalid session id")
    return config.sessions_dir() / f"{sid}.json"


def create(title: str = "New book") -> dict:
    config.ensure_dirs()
    s = {"id": uuid.uuid4().hex[:12], "title": title, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "messages": []}
    save(s)
    return s


def load(sid: str) -> dict | None:
    try:
        return json.loads(_p(sid).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save(s: dict) -> None:
    s["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _p(s["id"]).write_text(json.dumps(s, ensure_ascii=False))


def list_all() -> list[dict]:
    out = []
    for p in config.sessions_dir().glob("*.json"):
        try:
            s = json.loads(p.read_text())
            out.append({"id": s["id"], "title": s["title"], "updated": s["updated"], "n": len(s.get("messages", []))})
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(out, key=lambda x: x["updated"], reverse=True)


def delete(sid: str) -> bool:
    p = _p(sid)
    if p.exists():
        p.unlink()
        return True
    return False
