"""Session cookies and the Google sign-in flow. No vendor SDK: the OAuth code
exchange and the userinfo lookup are two plain REST calls via httpx, same
philosophy as app/providers/* — no client-library version conflicts, and the
same TRANSPORT test seam."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import urllib.parse

import httpx

from . import config
from .providers import base

COOKIE = "studio_session"
STATE_COOKIE = "studio_oauth_state"
MAX_AGE = 60 * 60 * 24 * 90  # 90 days

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


def _sign(payload: str) -> str:
    return hmac.new(config.session_secret().encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_cookie(user_id: str) -> str:
    payload = f"{user_id}.{int(time.time()) + MAX_AGE}"
    return f"{payload}.{_sign(payload)}"


def verify_cookie(token: str) -> str | None:
    parts = (token or "").split(".")
    if len(parts) != 3:
        return None
    user_id, exp, sig = parts
    if not hmac.compare_digest(_sign(f"{user_id}.{exp}"), sig):
        return None
    if not exp.isdigit() or int(exp) < time.time():
        return None
    return user_id


def new_state() -> str:
    return secrets.token_urlsafe(24)


def authorize_url(state: str, redirect_uri: str) -> str:
    params = {
        "client_id": config.GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


async def exchange_code(code: str, redirect_uri: str) -> dict:
    async with httpx.AsyncClient(transport=base.TRANSPORT, timeout=15) as c:
        r = await c.post(TOKEN_URL, data={
            "client_id": config.GOOGLE_CLIENT_ID, "client_secret": config.GOOGLE_CLIENT_SECRET,
            "code": code, "redirect_uri": redirect_uri, "grant_type": "authorization_code",
        })
    if r.status_code >= 400:
        raise ValueError(f"Google token exchange failed: {r.status_code} {r.text[:300]}")
    return r.json()


async def fetch_userinfo(access_token: str) -> dict:
    async with httpx.AsyncClient(transport=base.TRANSPORT, timeout=15) as c:
        r = await c.get(USERINFO_URL, headers={"authorization": f"Bearer {access_token}"})
    if r.status_code >= 400:
        raise ValueError(f"Google userinfo lookup failed: {r.status_code} {r.text[:300]}")
    return r.json()
