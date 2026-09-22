"""Central configuration. Everything is driven by environment variables so the
app deploys cleanly to any container host. API keys are read from the
environment only, never stored on disk or sent to the browser."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
IMAGES_DIR = DATA_DIR / "images"
SESSIONS_DIR = DATA_DIR / "sessions"
LIBRARY_DIR = DATA_DIR / "library"
SETTINGS_PATH = DATA_DIR / "settings.json"

STUDIO_PASSWORD = os.getenv("STUDIO_PASSWORD", "")
MAX_IMAGES_PER_TURN = int(os.getenv("MAX_IMAGES_PER_TURN", "6"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "10"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

# The three jobs you can assign to different providers.
ROLES = ("brain", "critic", "image")

_lock = threading.Lock()


def ensure_dirs() -> None:
    for d in (DATA_DIR, IMAGES_DIR, SESSIONS_DIR, LIBRARY_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    with _lock:
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}


def save_settings(settings: dict) -> None:
    ensure_dirs()
    with _lock:
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
