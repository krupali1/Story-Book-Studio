"""Central configuration. Everything is driven by environment variables so the
app deploys cleanly to any container host. API keys are read from the
environment only, never stored on disk or sent to the browser.

Storage is per-user: every user (a real Google account, or the implicit
`local` user when Google sign-in isn't configured) gets its own directory
under USERS_DIR. `set_current_user()` is called once per request (by
`require_auth`) and every path helper below resolves against whichever user
is current for that request. It's a contextvar, not a plain module
attribute, specifically so concurrent requests from different users can
never see each other's paths."""
from __future__ import annotations

import contextvars
import json
import os
import secrets
import threading
from pathlib import Path

BASE_DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
USERS_DIR = BASE_DATA_DIR / "users"

# The single implicit user when Google sign-in isn't configured — keeps the
# app fully clickable with zero setup (local dev, tests, the "demo mode"
# promise in the README) instead of forcing OAuth just to try it out.
LOCAL_USER_ID = "local"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "")  # optional override; derived from the request otherwise

MAX_IMAGES_PER_TURN = int(os.getenv("MAX_IMAGES_PER_TURN", "6"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "10"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

# The three jobs you can assign to different providers.
ROLES = ("brain", "critic", "image")

_lock = threading.Lock()
_current_user: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_user", default=None)


def google_configured() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


def set_current_user(user_id: str | None) -> None:
    _current_user.set(user_id)


def current_user() -> str:
    uid = _current_user.get()
    if not uid:
        raise RuntimeError("no current user set for this request — call set_current_user() first")
    return uid


def data_dir() -> Path:
    return USERS_DIR / current_user()


def images_dir() -> Path:
    return data_dir() / "images"


def sessions_dir() -> Path:
    return data_dir() / "sessions"


def library_dir() -> Path:
    return data_dir() / "library"


def settings_path() -> Path:
    return data_dir() / "settings.json"


def ensure_dirs() -> None:
    for d in (data_dir(), images_dir(), sessions_dir(), library_dir()):
        d.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    with _lock:
        try:
            return json.loads(settings_path().read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}


def save_settings(settings: dict) -> None:
    ensure_dirs()
    with _lock:
        settings_path().write_text(json.dumps(settings, indent=2))


# ---- session-cookie signing key -------------------------------------------
# Lazy and cached, never touched at import time: computing it eagerly would
# write a stray file the moment `app.config` is imported (e.g. during test
# collection), before a test has had a chance to redirect BASE_DATA_DIR.
_session_secret_cache: str | None = None


def session_secret() -> str:
    global _session_secret_cache
    if _session_secret_cache is not None:
        return _session_secret_cache
    env = os.getenv("SESSION_SECRET", "").strip()
    if env:
        _session_secret_cache = env
        return _session_secret_cache
    p = BASE_DATA_DIR / ".session_secret"
    try:
        _session_secret_cache = p.read_text().strip()
        return _session_secret_cache
    except FileNotFoundError:
        pass
    BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _session_secret_cache = secrets.token_hex(32)
    p.write_text(_session_secret_cache)
    return _session_secret_cache
