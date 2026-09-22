import pytest

from app import config
from app.providers import base

PROVIDER_ENV = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "STABILITY_API_KEY",
                "COMPAT_BASE_URL", "COMPAT_API_KEY", "COMPAT_MODEL", "BRAIN_PROVIDER", "BRAIN_MODEL",
                "CRITIC_PROVIDER", "CRITIC_MODEL", "IMAGE_PROVIDER", "IMAGE_MODEL"]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Fresh data dir, no provider keys, no Google sign-in (so require_auth falls
    back to the implicit local user, same zero-config behavior as before), for
    every test."""
    monkeypatch.setattr(config, "BASE_DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "USERS_DIR", tmp_path / "users")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "")
    monkeypatch.setattr(config, "_session_secret_cache", "test-session-secret", raising=False)
    for k in PROVIDER_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(base, "TRANSPORT", None)
    config.set_current_user(config.LOCAL_USER_ID)
    config.ensure_dirs()
    yield
    config.set_current_user(None)
