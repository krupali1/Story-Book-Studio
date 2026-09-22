import asyncio

import pytest

from app import images, library, tools
from app.tools import Ctx


def run(coro):
    return asyncio.run(coro)


def call(name, args, ctx=None):
    return run(tools.run(ctx or Ctx("s1", 6), {"tool": name, "args": args}))


def test_asset_versioning_and_search():
    a = library.save_asset("character", "Pip the Fox", "shy fox cub, orange scarf", ["fox", "Night"], {"bible": "x"})
    assert a["version"] == 1 and a["slug"] == "pip-the-fox" and a["tags"] == ["fox", "night"]
    b = library.save_asset("character", "Pip the Fox", "updated", ["fox"], {"bible": "y"})
    assert b["version"] == 2 and b["created"] == a["created"]
    library.save_asset("palette", "Moon blues", "deep blue night", ["night"], {"colors": ["#112244"]})
    assert [r["name"] for r in library.search("fox")] == ["Pip the Fox"]
    assert {r["name"] for r in library.search("night")} == {"Moon blues"}
    assert library.search("nothing-matches-this") == []
    assert len(library.search("")) == 2
    assert library.delete_asset("palette", "moon-blues") and not library.get_asset("palette", "moon-blues")


def test_bad_type_rejected():
    with pytest.raises(ValueError):
        library.save_asset("dragon", "x")


def test_taste_limits_and_history():
    library.set_taste("# Taste\n- likes soft gouache")
    library.set_taste("# Taste\n- likes paper cut")
    assert "paper cut" in library.get_taste()
    hist = list((library.config.DATA_DIR / "taste_history").glob("*.md"))
    assert len(hist) == 1 and "gouache" in hist[0].read_text()
    with pytest.raises(ValueError):
        library.set_taste("x" * (library.TASTE_MAX + 1))
    with pytest.raises(ValueError):
        library.set_taste("   ")


def test_feedback_digest():
    library.log_feedback({"verdict": "rejected", "tags": ["Too Dark"], "note": "scary"})
    library.log_feedback({"verdict": "rejected", "tags": ["too dark"]})
    library.log_feedback({"verdict": "approved", "tags": ["love palette"]})
    d = library.feedback_digest()
    assert "too dark (2)" in d and "love palette (1)" in d


def test_image_id_validation_blocks_traversal():
    assert not images.valid_id("../../etc/passwd")
    assert not images.exists("img_../x")
    with pytest.raises(ValueError):
        images.path("../../x")


def test_generate_critique_palette_moodboard_save():
    ctx = Ctx("s1", 6)
    g1 = call("generate_image", {"prompt": "a fox cub", "purpose": "character_sheet", "aspect": "landscape"}, ctx)
    g2 = call("generate_image", {"prompt": "a fox under the moon", "aspect": "portrait"}, ctx)
    assert g1.ok and g2.ok and ctx.images_left == 4
    i1, i2 = g1.images[0], g2.images[0]
    assert images.exists(i1) and images.meta(i1)["purpose"] == "character_sheet"

    c = call("critique_image", {"image_id": i2, "reference_ids": [i1]}, ctx)
    assert c.ok and "VERDICT" in c.text

    p = call("extract_palette", {"image_id": i1, "n": 4}, ctx)
    assert p.ok and p.text.count("#") >= 2

    m = call("make_moodboard", {"image_ids": [i1, i2], "palette": ["#112244", "nope"], "title": "T"}, ctx)
    assert m.ok and images.exists(m.images[0])

    s = call("save_asset", {"type": "character", "name": "Pip", "description": "fox", "tags": ["fox"],
                            "data": {"bible": "orange fox"}, "image_ids": [i1]}, ctx)
    assert s.ok and library.get_asset("character", "pip")["image_ids"] == [i1]


def test_tool_errors_are_soft():
    assert not call("generate_image", {"prompt": ""}).ok
    assert not call("generate_image", {"prompt": "x", "reference_ids": ["img_deadbeef"]}).ok
    assert not call("generate_image", {"prompt": "x", "style": "ghost"}).ok
    assert not call("save_asset", {"type": "dragon", "name": "x"}).ok
    assert not call("save_asset", {"type": "style", "name": "x", "data": "notadict"}).ok
    assert not call("nope", {}).ok
    assert not call("critique_image", {"image_id": "img_00000000"}).ok
    assert not call("read_asset", {"type": "character", "name": "ghost"}).ok


def test_image_budget_enforced():
    ctx = Ctx("s1", 1)
    assert call("generate_image", {"prompt": "one"}, ctx).ok
    over = call("generate_image", {"prompt": "two"}, ctx)
    assert not over.ok and "budget" in over.text.lower()


def test_style_prompt_is_applied_to_generation():
    library.save_asset("style", "Soft gouache", "", [], {"style_prompt": "soft gouache, grainy paper",
                                                          "negative_prompt": "neon"})
    r = call("generate_image", {"prompt": "a hedgehog", "style": "Soft gouache"})
    assert r.ok
    meta = images.meta(r.images[0])
    assert "soft gouache" in meta["prompt"] and "neon" in meta["negative"]


def test_log_feedback_and_update_taste_tools():
    r = call("log_feedback", {"verdict": "rejected", "tags": ["too dark"], "note": "n"})
    assert r.ok and library.recent_feedback(1)[0]["tags"] == ["too dark"]
    assert call("update_taste", {"markdown": "# t\n- a"}).ok
    assert not call("update_taste", {"markdown": ""}).ok
