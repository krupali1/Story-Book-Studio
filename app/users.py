"""User registry — one record per person who has signed in. Plain JSON, same
storage philosophy as the rest of the app. Keyed by a provider-namespaced id
(`google_<sub>`) so a future second provider (Apple, GitHub...) can't collide
with it."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from . import config

_lock = threading.Lock()


def _path() -> Path:
    return config.BASE_DATA_DIR / "users.json"


def _load() -> dict:
    try:
        return json.loads(_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(users: dict) -> None:
    config.BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps(users, indent=2, ensure_ascii=False))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def get(user_id: str) -> dict | None:
    return _load().get(user_id)


def upsert_from_google(claims: dict) -> dict:
    """claims: Google's userinfo response (sub, email, name, picture)."""
    sub = claims.get("sub")
    if not sub:
        raise ValueError("Google userinfo response had no 'sub'")
    user_id = f"google_{sub}"
    with _lock:
        users = _load()
        rec = users.get(user_id, {"id": user_id, "provider": "google", "created": _now()})
        rec.update(email=claims.get("email", ""), name=claims.get("name") or claims.get("email") or "Someone",
                   picture=claims.get("picture", ""), updated=_now())
        users[user_id] = rec
        _save(users)
        return rec


def public(rec: dict) -> dict:
    """What's safe to show other users browsing shared items — no email."""
    return {"id": rec["id"], "name": rec.get("name") or "Someone", "picture": rec.get("picture", "")}
