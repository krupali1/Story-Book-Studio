"""File-based library: characters, styles, palettes and books, plus the taste
profile and the feedback log. Plain JSON/Markdown on disk so it is easy to
back up, diff and move (and so a print pipeline can read it later)."""
from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from pathlib import Path

from . import config

TYPES = ("character", "style", "palette", "book")
TASTE_MAX = 6000
DEFAULT_TASTE = (
    "# Taste profile\n\n"
    "_No lessons yet. After each book, the studio distils what you liked and rejected here._\n"
)
_lock = threading.RLock()


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:60] or "untitled"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _dir(type_: str) -> Path:
    if type_ not in TYPES:
        raise ValueError(f"type must be one of {', '.join(TYPES)}")
    return config.LIBRARY_DIR / f"{type_}s"


def save_asset(type_: str, name: str, description: str = "", tags: list[str] | None = None,
               data: dict | None = None, image_ids: list[str] | None = None) -> dict:
    with _lock:
        slug = slugify(name)
        d = _dir(type_) / slug
        d.mkdir(parents=True, exist_ok=True)
        p = d / "asset.json"
        old = json.loads(p.read_text()) if p.exists() else None
        asset = {
            "type": type_, "slug": slug, "name": name.strip() or slug, "description": description or "",
            "tags": sorted({t.strip().lower() for t in (tags or []) if t and t.strip()}),
            "data": data or {}, "image_ids": image_ids or [],
            "created": old["created"] if old else _now(), "updated": _now(),
            "version": (old["version"] + 1) if old else 1,
        }
        p.write_text(json.dumps(asset, indent=2, ensure_ascii=False))
        _reindex()
        return asset


def get_asset(type_: str, name_or_slug: str) -> dict | None:
    p = _dir(type_) / slugify(name_or_slug) / "asset.json"
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def delete_asset(type_: str, slug: str) -> bool:
    import shutil
    with _lock:
        d = _dir(type_) / slugify(slug)
        if d.exists():
            shutil.rmtree(d)
            _reindex()
            return True
        return False


def list_assets(type_: str | None = None) -> list[dict]:
    out: list[dict] = []
    for t in ([type_] if type_ else TYPES):
        base = _dir(t)
        if not base.exists():
            continue
        for p in base.glob("*/asset.json"):
            try:
                out.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                continue
    return sorted(out, key=lambda a: a.get("updated", ""), reverse=True)


def _reindex() -> None:
    idx = [{k: a[k] for k in ("type", "slug", "name", "description", "tags", "image_ids", "updated")}
           for a in list_assets()]
    (config.LIBRARY_DIR / "index.json").write_text(json.dumps(idx, indent=2, ensure_ascii=False))


def search(query: str, type_: str | None = None, limit: int = 8) -> list[dict]:
    tokens = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 1]
    assets = list_assets(type_)
    if not tokens:
        return assets[:limit]
    scored = []
    for a in assets:
        hay_name, hay_tags = a["name"].lower(), " ".join(a["tags"])
        hay_desc, hay_data = a["description"].lower(), json.dumps(a["data"]).lower()
        s = sum(3 * (t in hay_name) + 2 * (t in hay_tags) + (t in hay_desc) + 0.5 * (t in hay_data) for t in tokens)
        if s > 0:
            scored.append((s, a))
    return [a for _, a in sorted(scored, key=lambda x: -x[0])][:limit]


def digest(per_type: int = 12) -> str:
    """Compact library overview injected into the agent's system prompt."""
    lines = []
    for t in TYPES:
        items = list_assets(t)[:per_type]
        for a in items:
            tags = f" [{', '.join(a['tags'])}]" if a["tags"] else ""
            lines.append(f"- {t}: {a['name']}: {a['description'][:110]}{tags}")
    return "\n".join(lines) or "(library is empty: this is the first book)"


# ---- taste profile -------------------------------------------------------

def _taste_path() -> Path:
    return config.DATA_DIR / "taste.md"


def get_taste() -> str:
    try:
        return _taste_path().read_text()
    except FileNotFoundError:
        return DEFAULT_TASTE


def set_taste(text: str) -> None:
    text = (text or "").strip()
    if not text:
        raise ValueError("taste profile can't be empty")
    if len(text) > TASTE_MAX:
        raise ValueError(f"taste profile too long ({len(text)} > {TASTE_MAX} chars): condense it")
    with _lock:
        config.ensure_dirs()
        p = _taste_path()
        if p.exists():
            hist = config.DATA_DIR / "taste_history"
            hist.mkdir(exist_ok=True)
            (hist / f"{time.strftime('%Y%m%d-%H%M%S')}.md").write_text(p.read_text())
        p.write_text(text + "\n")


# ---- feedback log --------------------------------------------------------

def _fb_path() -> Path:
    return config.DATA_DIR / "feedback.jsonl"


def log_feedback(entry: dict) -> dict:
    entry = {**entry, "ts": _now(), "tags": [t.strip().lower() for t in entry.get("tags", []) if t and t.strip()]}
    with _lock:
        config.ensure_dirs()
        with _fb_path().open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def recent_feedback(n: int = 20) -> list[dict]:
    try:
        lines = _fb_path().read_text().splitlines()
    except FileNotFoundError:
        return []
    out = []
    for ln in lines[-n:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out[::-1]


def feedback_digest() -> str:
    rows = recent_feedback(200)
    if not rows:
        return "(no feedback logged yet)"
    good, bad = Counter(), Counter()
    for r in rows:
        (good if r.get("verdict") == "approved" else bad if r.get("verdict") == "rejected" else Counter()).update(r["tags"])
    fmt = lambda c: ", ".join(f"{k} ({v})" for k, v in c.most_common(8)) or "none yet"
    recent = "; ".join(f"{r.get('verdict')}: {r.get('note') or ', '.join(r['tags'])}"[:90] for r in rows[:6])
    return f"Most rejected for: {fmt(bad)}\nMost loved: {fmt(good)}\nLatest: {recent}"
