"""Image storage and small image utilities (palette extraction, moodboards)."""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import config

ID_RE = re.compile(r"^img_[0-9a-f]{8}$")


def valid_id(image_id: str) -> bool:
    return bool(ID_RE.match(image_id or ""))


def _png_path(image_id: str) -> Path:
    if not valid_id(image_id):
        raise ValueError(f"invalid image id: {image_id!r}")
    return config.images_dir() / f"{image_id}.png"


def save(data: bytes, meta: dict | None = None) -> str:
    config.ensure_dirs()
    im = Image.open(BytesIO(data))
    im.load()
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    image_id = "img_" + uuid.uuid4().hex[:8]
    im.save(_png_path(image_id), "PNG")
    record = dict(meta or {})
    record.update(id=image_id, width=im.width, height=im.height, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (config.images_dir() / f"{image_id}.json").write_text(json.dumps(record))
    return image_id


def exists(image_id: str) -> bool:
    return valid_id(image_id) and _png_path(image_id).exists()


def path(image_id: str) -> Path:
    return _png_path(image_id)


def read(image_id: str) -> bytes:
    return _png_path(image_id).read_bytes()


def meta(image_id: str) -> dict:
    try:
        return json.loads((config.images_dir() / f"{image_id}.json").read_text()) if valid_id(image_id) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def for_vision(image_id: str, max_side: int = 1024) -> bytes:
    """Downscaled JPEG: keeps token cost and upload size sane for critique calls."""
    im = Image.open(_png_path(image_id))
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def extract_palette(data: bytes, n: int = 6) -> list[str]:
    n = max(2, min(int(n), 12))
    im = Image.open(BytesIO(data)).convert("RGB")
    im.thumbnail((256, 256))
    q = im.quantize(colors=max(n * 2, 8), method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette() or []
    colors: list[tuple[int, int, int]] = []
    for _, idx in sorted(q.getcolors() or [], reverse=True):
        rgb = tuple(pal[idx * 3: idx * 3 + 3])
        if len(rgb) == 3 and all(math.dist(rgb, c) > 38 for c in colors):
            colors.append(rgb)  # type: ignore[arg-type]
        if len(colors) == n:
            break
    return ["#%02x%02x%02x" % c for c in colors]


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def moodboard(images: list[bytes], palette: list[str], title: str) -> bytes:
    """Composite up to 9 images plus palette swatches into one PNG."""
    images = images[:9]
    palette = [c for c in palette if re.fullmatch(r"#[0-9a-fA-F]{6}", c or "")][:8]
    W, pad, header = 1600, 36, 120
    cols = 3 if len(images) >= 3 else max(1, len(images))
    cell = (W - pad * (cols + 1)) // cols
    rows = max(1, math.ceil(len(images) / cols))
    footer = 190 if palette else 0
    H = header + rows * (cell + pad) + pad + footer
    ink, paper = (29, 35, 80), (238, 241, 246)
    canvas = Image.new("RGB", (W, H), paper)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, W, header - 20], fill=ink)
    d.text((pad, 34), (title or "Moodboard")[:60], fill=(255, 200, 61), font=_font(44))
    for i, raw in enumerate(images):
        im = Image.open(BytesIO(raw)).convert("RGB")
        im.thumbnail((cell, cell))
        x = pad + (i % cols) * (cell + pad) + (cell - im.width) // 2
        y = header + (i // cols) * (cell + pad) + (cell - im.height) // 2
        d.rectangle([x - 6, y - 6, x + im.width + 6, y + im.height + 6], fill=(255, 255, 255), outline=ink, width=3)
        canvas.paste(im, (x, y))
    if palette:
        y0 = H - footer + 20
        sw = min(150, (W - pad * 2) // len(palette) - 12)
        for i, hx in enumerate(palette):
            x = pad + i * (sw + 12)
            d.rectangle([x, y0, x + sw, y0 + 100], fill=hx, outline=ink, width=3)
            d.text((x, y0 + 110), hx, fill=ink, font=_font(24))
    buf = BytesIO()
    canvas.save(buf, "PNG")
    return buf.getvalue()
