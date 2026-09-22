"""FastAPI app: JSON API + SSE chat stream + static single-page UI."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import re
import zipfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import agent, config, images, library, registry, sessions

config.ensure_dirs()
app = FastAPI(title="Storybook Studio")
STATIC = Path(__file__).resolve().parent.parent / "static"
COOKIE = "studio_auth"


# ---------------------------------------------------------------- auth ----
def _token() -> str:
    return hmac.new(config.STUDIO_PASSWORD.encode(), b"storybook-studio-v1", hashlib.sha256).hexdigest()


def require_auth(request: Request) -> None:
    if config.STUDIO_PASSWORD and not hmac.compare_digest(request.cookies.get(COOKIE, ""), _token()):
        raise HTTPException(401, "Sign in required")


class LoginIn(BaseModel):
    password: str


@app.post("/api/login")
async def login(body: LoginIn, request: Request, response: Response):
    if not config.STUDIO_PASSWORD:
        return {"ok": True}
    if not hmac.compare_digest(body.password.encode(), config.STUDIO_PASSWORD.encode()):
        await asyncio.sleep(1.0)  # cheap brute-force brake
        raise HTTPException(401, "Wrong password")
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(COOKIE, _token(), httponly=True, samesite="lax", secure=secure, max_age=60 * 60 * 24 * 90)
    return {"ok": True}


@app.get("/api/me")
async def me(request: Request):
    required = bool(config.STUDIO_PASSWORD)
    authed = (not required) or hmac.compare_digest(request.cookies.get(COOKIE, ""), _token())
    return {"auth_required": required, "authed": authed}


@app.get("/health")
async def health():
    return {"ok": True}


# ------------------------------------------------------------ settings ----
@app.get("/api/settings", dependencies=[Depends(require_auth)])
async def get_settings():
    return registry.describe()


class RoleIn(BaseModel):
    provider: str
    model: str = ""


class SettingsIn(BaseModel):
    brain: RoleIn | None = None
    critic: RoleIn | None = None
    image: RoleIn | None = None


@app.put("/api/settings", dependencies=[Depends(require_auth)])
async def put_settings(body: SettingsIn):
    saved = config.load_settings()
    for role in config.ROLES:
        sel = getattr(body, role)
        if sel is None:
            continue
        specs = registry.IMAGE_SPECS if role == "image" else registry.LLM_SPECS
        if sel.provider not in specs:
            raise HTTPException(400, f"Unknown provider '{sel.provider}' for {role}")
        if not registry.configured(specs, sel.provider):
            keys = " or ".join(specs[sel.provider]["keys"])
            raise HTTPException(400, f"{specs[sel.provider]['label']} isn't configured on the server. Set {keys}.")
        saved[role] = {"provider": sel.provider, "model": sel.model.strip()}
    config.save_settings(saved)
    return registry.describe()


# ------------------------------------------------------------ sessions ----
@app.get("/api/sessions", dependencies=[Depends(require_auth)])
async def list_sessions():
    return sessions.list_all()


@app.post("/api/sessions", dependencies=[Depends(require_auth)])
async def create_session():
    return sessions.create()


@app.get("/api/sessions/{sid}", dependencies=[Depends(require_auth)])
async def get_session(sid: str):
    s = _session_or_404(sid)
    return s


@app.delete("/api/sessions/{sid}", dependencies=[Depends(require_auth)])
async def delete_session(sid: str):
    _session_or_404(sid)
    sessions.delete(sid)
    return {"ok": True}


def _session_or_404(sid: str) -> dict:
    try:
        s = sessions.load(sid)
    except ValueError:
        s = None
    if not s:
        raise HTTPException(404, "Session not found")
    return s


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    image_ids: list[str] = []
    explore: bool = False


_tasks: set[asyncio.Task] = set()


@app.post("/api/sessions/{sid}/chat", dependencies=[Depends(require_auth)])
async def chat(sid: str, body: ChatIn):
    _session_or_404(sid)
    queue: asyncio.Queue = asyncio.Queue()

    async def produce():
        try:
            async for ev in agent.run_turn(sid, body.text, body.image_ids, body.explore):
                await queue.put(ev)
        except Exception as e:  # never leave the stream hanging
            await queue.put({"type": "error", "message": f"Unexpected error: {e!r}"})
        finally:
            await queue.put(None)

    # The turn runs to completion even if the browser disconnects, so long image jobs still get saved.
    task = asyncio.create_task(produce())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)

    async def stream():
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": ping\n\n"  # keep proxies from closing idle streams during slow image calls
                continue
            if ev is None:
                return
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# -------------------------------------------------------------- images ----
@app.post("/api/upload", dependencies=[Depends(require_auth)])
async def upload(file: UploadFile = File(...)):
    data = await file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image too large (max {config.MAX_UPLOAD_MB} MB)")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(400, "That file isn't a readable image")
    return {"id": images.save(data, {"purpose": "upload", "prompt": file.filename or "upload"})}


@app.get("/api/images/{image_id}", dependencies=[Depends(require_auth)])
async def get_image(image_id: str):
    if not images.exists(image_id):
        raise HTTPException(404, "Image not found")
    return FileResponse(images.path(image_id), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=31536000, immutable"})


# ------------------------------------------------------------- library ----
@app.get("/api/library", dependencies=[Depends(require_auth)])
async def get_library(type: str | None = None):
    if type and type not in library.TYPES:
        raise HTTPException(400, "bad type")
    return library.list_assets(type)


@app.get("/api/library/{type}/{slug}", dependencies=[Depends(require_auth)])
async def get_asset(type: str, slug: str):
    a = library.get_asset(type, slug) if type in library.TYPES else None
    if not a:
        raise HTTPException(404, "Not found")
    return a


@app.delete("/api/library/{type}/{slug}", dependencies=[Depends(require_auth)])
async def delete_asset(type: str, slug: str):
    if type not in library.TYPES or not library.delete_asset(type, slug):
        raise HTTPException(404, "Not found")
    return {"ok": True}


@app.get("/api/books/{slug}/export", dependencies=[Depends(require_auth)])
async def export_book(slug: str):
    """ZIP with book.json, a readable story.md and the page images. This is the
    hand-off point for a future print pipeline."""
    book = library.get_asset("book", slug)
    if not book:
        raise HTTPException(404, "Book not found")
    buf = io.BytesIO()
    pages = (book.get("data") or {}).get("pages") or []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("book.json", json.dumps(book, indent=2, ensure_ascii=False))
        md = [f"# {book['name']}", "", book.get("description", ""), ""]
        for i, p in enumerate(pages, 1):
            md += [f"## Page {p.get('n', i)}", "", str(p.get("text", "")), ""]
            iid = p.get("image_id")
            if iid and images.exists(iid):
                z.write(images.path(iid), f"images/page-{int(p.get('n', i)) if str(p.get('n', i)).isdigit() else i:02d}.png")
        z.writestr("story.md", "\n".join(md))
        for iid in book.get("image_ids", []):
            if images.exists(iid):
                z.write(images.path(iid), f"images/{iid}.png")
    name = re.sub(r"[^a-z0-9-]", "", book["slug"]) or "book"
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})


# --------------------------------------------------------- taste & fb ----
class TasteIn(BaseModel):
    markdown: str


@app.get("/api/taste", dependencies=[Depends(require_auth)])
async def get_taste():
    return {"markdown": library.get_taste(), "max": library.TASTE_MAX}


@app.put("/api/taste", dependencies=[Depends(require_auth)])
async def put_taste(body: TasteIn):
    try:
        library.set_taste(body.markdown)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


class FeedbackIn(BaseModel):
    image_id: str | None = None
    verdict: str
    tags: list[str] = []
    note: str = ""
    session_id: str | None = None


@app.post("/api/feedback", dependencies=[Depends(require_auth)])
async def post_feedback(body: FeedbackIn):
    if body.verdict not in ("approved", "rejected", "note"):
        raise HTTPException(400, "verdict must be approved, rejected or note")
    entry = body.model_dump()
    if body.image_id and images.valid_id(body.image_id):
        entry["prompt"] = images.meta(body.image_id).get("prompt", "")[:300]
    return library.log_feedback(entry)


@app.get("/api/feedback", dependencies=[Depends(require_auth)])
async def get_feedback():
    return library.recent_feedback(60)


# -------------------------------------------------------------- static ----
@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
