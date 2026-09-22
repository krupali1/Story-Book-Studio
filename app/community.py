"""Cross-user browsing of assets people have explicitly marked shared. Reads
other users' library index files directly, read-only, and only ever the
`shared: true` subset — never anything else in someone else's private data,
and never their email (see users.public())."""
from __future__ import annotations

import json
import re

from . import config, library, users

_SAFE_ID = re.compile(r"^[A-Za-z0-9_]+$")


def _user_dir(owner_id: str):
    if not _SAFE_ID.match(owner_id or ""):
        return None
    d = config.USERS_DIR / owner_id
    return d if d.is_dir() else None


def list_shared(limit: int = 60) -> list[dict]:
    out: list[dict] = []
    if not config.USERS_DIR.exists():
        return out
    for d in config.USERS_DIR.iterdir():
        if not d.is_dir():
            continue
        try:
            idx = json.loads((d / "library" / "index.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        rec = users.get(d.name)
        owner = users.public(rec) if rec else {"id": d.name, "name": "Someone", "picture": ""}
        for a in idx:
            if a.get("shared"):
                out.append({**a, "owner": owner})
    out.sort(key=lambda a: a.get("updated", ""), reverse=True)
    return out[:limit]


def get_shared(owner_id: str, type_: str, slug: str) -> dict | None:
    if type_ not in library.TYPES:
        return None
    d = _user_dir(owner_id)
    if not d:
        return None
    try:
        a = json.loads((d / "library" / f"{type_}s" / library.slugify(slug) / "asset.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not a.get("shared"):
        return None
    rec = users.get(owner_id)
    a["owner"] = users.public(rec) if rec else {"id": owner_id, "name": "Someone", "picture": ""}
    return a


def shared_image_owner_ok(owner_id: str, image_id: str) -> bool:
    """True only if some asset this owner has explicitly shared references this image."""
    d = _user_dir(owner_id)
    if not d:
        return False
    try:
        idx = json.loads((d / "library" / "index.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return any(a.get("shared") and image_id in (a.get("image_ids") or []) for a in idx)


def shared_image_path(owner_id: str, image_id: str):
    d = _user_dir(owner_id)
    if not d or not shared_image_owner_ok(owner_id, image_id):
        return None
    p = d / "images" / f"{image_id}.png"
    return p if p.exists() else None
