import pytest

from app import config
from app.providers import base

PROVIDER_ENV = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "STABILITY_API_KEY",
                "COMPAT_BASE_URL", "COMPAT_API_KEY", "COMPAT_MODEL", "BRAIN_PROVIDER", "BRAIN_MODEL",
                "CRITIC_PROVIDER", "CRITIC_MODEL", "IMAGE_PROVIDER", "IMAGE_MODEL"]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Fresh data dir, no provider keys, no auth, for every test."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "STUDIO_PASSWORD", "")
    for k in PROVIDER_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(base, "TRANSPORT", None)
    config.ensure_dirs()
    yield
