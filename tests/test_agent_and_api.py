import io
import json
import zipfile

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import agent, config, images, library, registry, sessions
from app.main import app
from app.providers import base
from app.providers.base import ProviderError


@pytest.fixture
def client():
    return TestClient(app)


def sse(resp):
    events = []
    for chunk in resp.text.split("\n\n"):
        if chunk.startswith("data: "):
            events.append(json.loads(chunk[6:]))
    return events


def png_bytes():
    b = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 120, 200)).save(b, "PNG")
    return b.getvalue()


def test_parse_tool_calls():
    txt, calls, err = agent.parse_tool_calls('Hi\n```tool_calls\n[{"tool":"a","args":{}}]\n```')
    assert txt == "Hi" and calls == [{"tool": "a", "args": {}}] and err is None
    assert agent.parse_tool_calls("plain")[1:] == ([], None)
    assert agent.parse_tool_calls('```tool_calls\n{"tool":"a"}\n```')[1] == [{"tool": "a"}]  # single object tolerated
    assert "valid JSON" in agent.parse_tool_calls("```tool_calls\n[{oops}]\n```")[2]
    assert "cut off" in agent.parse_tool_calls('x\n```tool_calls\n[{"tool":"a"')[2]
    assert agent.parse_tool_calls("```tool_calls\n[1,2]\n```")[2]


def test_full_demo_turn_over_http(client):
    s = client.post("/api/sessions").json()
    r = client.post(f"/api/sessions/{s['id']}/chat", json={"text": "A fox who fears the dark, age 4"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = sse(r)
    kinds = [e["type"] for e in ev]
    assert kinds[-1] == "done" and "error" not in kinds and "title" in kinds
    tools_run = [e["tool"] for e in ev if e["type"] == "tool_call"]
    assert tools_run == ["generate_image", "generate_image", "extract_palette", "critique_image",
                         "make_moodboard", "save_asset", "log_feedback"]
    assert all(e["ok"] for e in ev if e["type"] == "tool_result")
    shown = [i for e in ev if e["type"] == "tool_result" for i in e["images"]]
    assert len(shown) >= 3 and all(client.get(f"/api/images/{i}").status_code == 200 for i in shown)

    # persisted session replays with kinds the UI understands
    saved = client.get(f"/api/sessions/{s['id']}").json()
    assert saved["title"].startswith("A fox") and {m["kind"] for m in saved["messages"]} == {"user", "assistant", "tools"}
    # library + feedback were written by the agent's tool calls
    assert [a["name"] for a in client.get("/api/library").json()] == ["Demo moon palette"]
    assert client.get("/api/feedback").json()[0]["tags"] == ["love palette"]
    # and the next turn's system prompt carries that learning
    sys_prompt = __import__("app.prompts", fromlist=["x"]).build_system()
    assert "Demo moon palette" in sys_prompt and "love palette" in sys_prompt


def test_session_lifecycle_and_404s(client):
    a = client.post("/api/sessions").json()
    assert client.get("/api/sessions").json()[0]["n"] == 0
    assert client.delete(f"/api/sessions/{a['id']}").json() == {"ok": True}
    assert client.get(f"/api/sessions/{a['id']}").status_code == 404
    assert client.get("/api/sessions/not-an-id").status_code == 404
    assert client.post("/api/sessions/aaaaaaaaaaaa/chat", json={"text": "x"}).status_code == 404
    assert client.get("/api/images/img_../etc").status_code in (404, 422)
    assert client.get("/api/images/img_00000000").status_code == 404


def test_settings_roundtrip_and_validation(client, monkeypatch):
    d = client.get("/api/settings").json()
    assert d["demo_mode"] and {p["id"] for p in d["llm"]} >= {"anthropic", "openai", "gemini", "mock"}
    assert "KEY" not in json.dumps(d).replace("_API_KEY", "").replace("ANTHROPIC_", "")  # env NAMES only, never values
    # cannot select a provider that has no key on the server
    r = client.put("/api/settings", json={"brain": {"provider": "anthropic", "model": ""}})
    assert r.status_code == 400 and "ANTHROPIC_API_KEY" in r.json()["detail"]
    assert client.put("/api/settings", json={"image": {"provider": "nope"}}).status_code == 400
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value")
    monkeypatch.setenv("GEMINI_API_KEY", "g-test-value")
    ok = client.put("/api/settings", json={"brain": {"provider": "openai", "model": "gpt-5-mini"},
                                           "critic": {"provider": "gemini", "model": "gemini-2.5-flash"},
                                           "image": {"provider": "gemini", "model": ""}}).json()
    assert ok["roles"]["brain"]["model"] == "gpt-5-mini" and ok["roles"]["critic"]["provider"] == "gemini"
    assert ok["roles"]["image"]["model"] == "gemini-2.5-flash-image" and not ok["demo_mode"]
    assert "sk-test-value" not in json.dumps(ok) and "g-test-value" not in json.dumps(ok)


def test_no_google_configured_is_the_zero_config_demo(client):
    """Default test setup: no GOOGLE_CLIENT_ID/SECRET, so there's one implicit
    local user and no login wall — same as before this feature existed."""
    assert client.get("/api/me").json() == {"auth_required": False, "authed": True, "user": None}
    assert client.get("/api/sessions").status_code == 200
    assert client.get("/auth/google/login").status_code == 404


def _mock_google(sub, name, email):
    def handler(req):
        if "oauth2.googleapis.com/token" in str(req.url):
            return httpx.Response(200, json={"access_token": f"tok-{sub}"})
        return httpx.Response(200, json={"sub": sub, "email": email, "name": name, "picture": ""})
    base.TRANSPORT = httpx.MockTransport(handler)


def _login_as(client, sub, name, email):
    _mock_google(sub, name, email)
    r = client.get("/auth/google/login", follow_redirects=False)
    state = r.cookies.get("studio_oauth_state")
    cb = client.get("/auth/google/callback", params={"code": "c", "state": state}, follow_redirects=False)
    assert cb.status_code in (302, 307) and cb.headers["location"] == "/"
    return cb


def test_google_auth_flow(client, monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "test-client-secret")
    assert client.get("/api/me").json() == {"auth_required": True, "authed": False, "user": None}
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/health").status_code == 200 and client.get("/").status_code == 200

    r = client.get("/auth/google/login", follow_redirects=False)
    assert r.status_code in (302, 307) and "accounts.google.com" in r.headers["location"]
    assert r.cookies.get("studio_oauth_state")

    # wrong state is rejected
    bad = client.get("/auth/google/callback", params={"code": "c", "state": "not-it"}, follow_redirects=False)
    assert bad.headers["location"] == "/?auth_error=state_mismatch"

    _login_as(client, "111", "Reader", "reader@example.com")
    me = client.get("/api/me").json()
    assert me == {"auth_required": True, "authed": True,
                  "user": {"name": "Reader", "email": "reader@example.com", "picture": ""}}
    assert client.get("/api/sessions").status_code == 200

    client.post("/auth/logout")
    assert client.get("/api/me").json()["authed"] is False
    assert client.get("/api/sessions").status_code == 401


def test_per_user_isolation_and_sharing(client, monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "test-client-secret")

    _login_as(client, "111", "Alice", "alice@example.com")
    s = client.post("/api/sessions").json()
    client.post(f"/api/sessions/{s['id']}/chat", json={"text": "A fox who fears the dark, age 4"})
    lib = client.get("/api/library").json()
    palette = next(a for a in lib if a["type"] == "palette")
    assert client.put(f"/api/library/palette/{palette['slug']}/share", json={"shared": True}).status_code == 200

    _login_as(client, "222", "Bob", "bob@example.com")
    # Bob is a brand-new user — Alice's library is invisible to him directly...
    assert client.get("/api/library").json() == []
    # ...but her one shared palette shows up in the community gallery, with no email leaked.
    shared = client.get("/api/community").json()
    assert len(shared) == 1
    assert shared[0]["name"] == palette["name"] and shared[0]["owner"] == {"id": "google_111", "name": "Alice", "picture": ""}
    detail = client.get(f"/api/community/google_111/palette/{palette['slug']}").json()
    assert detail["slug"] == palette["slug"]
    assert client.get("/api/community/google_111/palette/not-a-real-slug").status_code == 404
    assert client.get("/api/community/google_222/palette/" + palette["slug"]).status_code == 404  # wrong owner
    # the shared asset's own image is fetchable by anyone (route order matters here:
    # /images/{id} must not be shadowed by the /{type}/{slug} route above it)
    img_id = palette["image_ids"][0]
    img_resp = client.get(f"/api/community/google_111/images/{img_id}")
    assert img_resp.status_code == 200 and img_resp.headers["content-type"] == "image/png"
    assert client.get(f"/api/community/google_222/images/{img_id}").status_code == 404  # wrong owner
    assert client.get("/api/community/google_111/images/img_00000000").status_code == 404  # not referenced by any shared asset

    # Alice can un-share it and it disappears from the gallery
    _login_as(client, "111", "Alice", "alice@example.com")
    client.put(f"/api/library/palette/{palette['slug']}/share", json={"shared": False})
    _login_as(client, "222", "Bob", "bob@example.com")
    assert client.get("/api/community").json() == []


def test_upload_and_reference_flow(client):
    up = client.post("/api/upload", files={"file": ("ref.png", png_bytes(), "image/png")}).json()
    assert images.exists(up["id"])
    bad = client.post("/api/upload", files={"file": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 400
    s = client.post("/api/sessions").json()
    ev = sse(client.post(f"/api/sessions/{s['id']}/chat", json={"text": "use my sketch", "image_ids": [up["id"], "img_bogus000"]}))
    assert ev[-1]["type"] == "done"
    saved = client.get(f"/api/sessions/{s['id']}").json()
    assert saved["messages"][0]["image_ids"] == [up["id"]]  # unknown ids dropped
    msgs = agent.to_llm_messages(saved)
    # the attachment stays visible even after a multi-step tool turn appended many messages
    assert len(saved["messages"]) > 6
    assert up["id"] in msgs[0].text and len(msgs[0].images) == 1


def test_only_recent_user_turns_carry_image_pixels():
    ids = [images.save(png_bytes(), {}) for _ in range(3)]
    sess = {"messages": []}
    for n, i in enumerate(ids):
        sess["messages"] += [{"kind": "user", "text": f"turn {n}", "image_ids": [i]},
                             {"kind": "assistant", "text": "ok", "raw": "ok"}]
    msgs = [m for m in agent.to_llm_messages(sess) if m.role == "user"]
    assert [len(m.images) for m in msgs] == [0, 1, 1]
    assert ids[0] in msgs[0].text  # older attachment is still referenced by id, just not re-sent as pixels


def test_taste_feedback_library_export(client):
    assert "No lessons yet" in client.get("/api/taste").json()["markdown"]
    assert client.put("/api/taste", json={"markdown": "# Taste\n- soft gouache"}).status_code == 200
    assert client.put("/api/taste", json={"markdown": "x" * 7000}).status_code == 400
    assert client.post("/api/feedback", json={"verdict": "meh"}).status_code == 400
    assert client.post("/api/feedback", json={"verdict": "rejected", "tags": ["too dark"]}).status_code == 200

    i1 = images.save(png_bytes(), {"prompt": "p"})
    i2 = images.save(png_bytes(), {"prompt": "q"})
    library.save_asset("book", "Fox and the Moon", "a bedtime tale", ["fox"], {
        "pages": [{"n": 1, "text": "Once upon a moon.", "image_id": i1}, {"n": 2, "text": "The end.", "image_id": i2}]}, [i1])
    z = zipfile.ZipFile(io.BytesIO(client.get("/api/books/fox-and-the-moon/export").content))
    names = set(z.namelist())
    assert {"book.json", "story.md", "images/page-01.png", "images/page-02.png"} <= names
    assert "Once upon a moon." in z.read("story.md").decode()
    assert client.get("/api/books/ghost/export").status_code == 404
    assert client.get("/api/library/book/fox-and-the-moon").json()["version"] == 1
    assert client.delete("/api/library/book/fox-and-the-moon").json() == {"ok": True}
    assert client.get("/api/library/dragon/x").status_code == 404


# ---- agent robustness with a scripted brain -------------------------------
class ScriptedBrain:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def complete(self, system, messages, max_tokens=8000):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def turn(monkeypatch, brain, text="go"):
    monkeypatch.setattr(registry, "build_llm", lambda role: brain)
    s = sessions.create()

    async def collect():
        return [e async for e in agent.run_turn(s["id"], text, [])]

    import asyncio
    return s, asyncio.run(collect())


def test_brain_repairs_malformed_tool_block(monkeypatch):
    brain = ScriptedBrain(["Trying.\n```tool_calls\n[{bad json}]\n```",
                           'Retry.\n```tool_calls\n[{"tool":"search_library","args":{"query":""}}]\n```', "All done."])
    s, ev = turn(monkeypatch, brain)
    assert [e["type"] for e in ev][-1] == "done" and len(brain.calls) == 3
    # the model was told what was wrong on the second call
    assert "not valid JSON" in brain.calls[1][-1].text
    assert [e["tool"] for e in ev if e["type"] == "tool_call"] == ["search_library"]


def test_unknown_tool_and_call_cap_and_step_limit(monkeypatch):
    many = json.dumps([{"tool": "search_library", "args": {}}] * 6)
    brain = ScriptedBrain([f"```tool_calls\n{many}\n```", "fine"])
    _, ev = turn(monkeypatch, brain)
    assert sum(e["type"] == "tool_call" for e in ev) == agent.MAX_CALLS_PER_STEP
    assert "Only the first" in brain.calls[1][-1].text

    monkeypatch.setattr(config, "MAX_AGENT_STEPS", 3)
    loop = '```tool_calls\n[{"tool":"nope","args":{}}]\n```'
    brain = ScriptedBrain([loop] * 3)
    _, ev = turn(monkeypatch, brain)
    assert any("step limit" in e.get("text", "") for e in ev) and ev[-1]["type"] == "done"


def test_provider_failure_surfaces_as_error_event(monkeypatch):
    _, ev = turn(monkeypatch, ScriptedBrain([ProviderError("Anthropic returned 401: bad key")]))
    assert ev[-1] == {"type": "error", "message": "Anthropic returned 401: bad key"}


def test_image_failure_is_reported_to_the_brain_not_fatal(monkeypatch):
    class Boom:
        supports_reference = True
        async def generate(self, *a, **k):
            raise ProviderError("OpenAI Images returned 400: content policy")

    monkeypatch.setattr(registry, "build_image", lambda: Boom())
    brain = ScriptedBrain(['```tool_calls\n[{"tool":"generate_image","args":{"prompt":"x"}}]\n```', "Sorry, that failed."])
    _, ev = turn(monkeypatch, brain)
    res = [e for e in ev if e["type"] == "tool_result"][0]
    assert res["ok"] is False and "content policy" in brain.calls[1][-1].text and ev[-1]["type"] == "done"


def test_reference_dropped_with_note_when_provider_cannot_take_them(monkeypatch):
    seen = {}

    class NoRef:
        supports_reference = False
        async def generate(self, prompt, negative, aspect, references):
            seen["refs"] = references
            return png_bytes()

    from app import tools
    ref = images.save(png_bytes(), {})
    monkeypatch.setattr(registry, "build_image", lambda: NoRef())
    import asyncio
    r = asyncio.run(tools.run(tools.Ctx("s", 3), {"tool": "generate_image", "args": {"prompt": "p", "reference_ids": [ref]}}))
    assert r.ok and seen["refs"] == [] and "cannot take reference images" in r.text
