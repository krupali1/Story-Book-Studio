# Storybook Studio: project context for Claude Code

A self-hosted AI design agent for illustrated children's picture books. The user chats with it; it plans,
writes prompts, generates and critiques illustrations, and keeps a persistent library plus a taste profile
that improves across books. Personal project (not client work). Owner: Ankit.

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q                    # 37 tests, no network or API keys needed
uvicorn app.main:app --reload --port 8000
docker compose up --build              # needs .env (copy .env.example)
```

With no API keys the app runs in **demo mode** (MockLLM + MockImage), so the UI is fully clickable offline.
With no `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` set, there's also no login wall: everything runs as one
implicit `local` user, same as before multi-user existed.

## Architecture in one paragraph

FastAPI backend + one vanilla-JS file (`static/index.html`, no build step). Three independently selectable AI
**roles**: `brain` (chat/planning/prompts, drives tools), `critic` (vision review of generated images),
`image` (illustration generation). Provider+model per role is chosen in the Providers dialog and saved to
`<user>/settings.json`. API keys come **only** from server env vars and are never sent to the browser
(`registry.describe()` exposes env var *names*, never values). Storage is plain JSON/Markdown/PNG files, one
directory tree per user under `DATA_DIR/users/<user_id>/`; there is no database. Because of that, run exactly
**one** worker. Multi-user: Google sign-in (see below) gates access and each signed-in person gets their own
private directory; an asset only leaves it if its owner explicitly marks it `shared`, in which case it's
readable (never writable) by anyone else via the Community endpoints.

## Code map

- `app/config.py`: env config, per-user path helpers (`data_dir()`, `images_dir()`, ... — all resolve against
  a **contextvar**, `set_current_user()`/`current_user()`, not a fixed constant), `load_settings/save_settings`,
  `ROLES`, `session_secret()` (lazy, auto-generated and persisted to `DATA_DIR/.session_secret` if unset — never
  computed at import time, so importing the module never has a disk side effect).
- `app/auth.py`: session-cookie signing (HMAC, `user_id.expiry.sig`) and the Google OAuth code exchange +
  userinfo lookup, both plain `httpx` calls (no vendor SDK), through the same `providers/base.TRANSPORT` test
  seam as the provider adapters.
- `app/users.py`: the user registry (`DATA_DIR/users.json`), keyed `google_<sub>`. `public()` strips email
  before an owner's identity is shown to anyone browsing their shared items.
- `app/community.py`: read-only cross-user access to `shared: true` library assets and the images they
  reference — nothing else in another user's directory is ever reachable.
- `app/registry.py`: provider catalogue and `resolve_role()`. Precedence: saved Settings > `{ROLE}_PROVIDER/MODEL`
  env > first configured provider > mock. Critic follows brain unless set. `fallback=True` if a saved provider
  lost its key. `build_llm(role)` / `build_image()`.
- `app/providers/base.py`: `Msg`, `normalize()`, `post()` with retry (network/429/5xx). Module-level `TRANSPORT`
  is the test seam for mocking HTTP.
- `app/providers/llm.py`: `AnthropicLLM`, `OpenAILLM` (`compat=True` uses `max_tokens`), `GeminiLLM`, `MockLLM`.
  All expose `async complete(system, messages, max_tokens) -> str`.
- `app/providers/image.py`: `OpenAIImage`, `GeminiImage`, `StabilityImage` (no reference images), `MockImage`.
  All expose `async generate(prompt, negative, aspect, references) -> bytes` and `supports_reference`.
- `app/tools.py`: the agent's tools (`search_library`, `read_asset`, `generate_image`, `critique_image`,
  `extract_palette`, `make_moodboard`, `save_asset`, `log_feedback`, `update_taste`). Registered with `@tool`;
  `run()` never raises, failures come back as soft `ToolResult(ok=False)`.
- `app/agent.py`: the loop. `run_turn()` is an async generator of UI events. Tool protocol is a fenced
  ```` ```tool_calls ```` JSON block in the model's text (provider-agnostic, deliberately **not** native
  function calling, so every provider works). `parse_tool_calls()` tolerates malformed/truncated blocks and the
  model is told to repair them. Caps: `MAX_AGENT_STEPS`, 4 tool calls per round, `MAX_IMAGES_PER_TURN`.
- `app/prompts.py`: `build_system()` injects live provider labels, taste profile, library digest, feedback digest.
- `app/library.py`: assets (character/style/palette/book), each with a `shared` flag (`set_shared()` flips it
  without bumping `version` — visibility, not an edit), versioned taste profile, JSONL feedback log.
- `app/images.py`: image store (`img_XXXXXXXX` ids, path-traversal safe), vision downscale, palette, moodboard.
- `app/main.py`: routes, `require_auth` (Google sign-in when configured, else the implicit `local` user — **must
  stay `async def`**: FastAPI runs sync dependencies in a threadpool, and a contextvar `.set()` made there
  doesn't propagate back to the request's own task, so every other route would silently see no current user),
  `/auth/google/{login,callback}`, `/auth/logout`, the sharing/community routes (note the community image route
  is declared *above* the `{type}/{slug}` route — both match a 3-segment path, so order decides which wins), SSE
  chat endpoint that runs the turn as a detached task so it survives client disconnects, and
  `/api/books/{slug}/export`.

## Design decisions worth preserving

1. **Text tool protocol, not native function calling**, so any provider (including Ollama) can be the brain.
2. **Only the last `VISION_WINDOW` (6) chat messages carry image bytes**, to control vision cost. Reference images
   passed to `generate_image` via `reference_ids` are re-read from disk, so they bypass this window.
3. **The prompt restates the character description on every image prompt**, because some providers ignore
   reference images. Character consistency must not depend on refs alone.
4. **Taste profile and feedback log are the learning mechanism** and are fed back into every system prompt.
   Keep/reject actions in the UI write tagged feedback.
5. **Child-safety and no copyrighted characters** are hard rules in the system prompt. Do not weaken them.
6. **The book export ZIP (`book.json` + `story.md` + page images) is the intended hand-off to a future print
   pipeline** (print-ready PDF/bleed/ICC). That pipeline does not exist yet.

## Testing conventions

`tests/conftest.py` isolates `BASE_DATA_DIR`/`USERS_DIR`, clears all provider env vars and Google credentials,
resets `base.TRANSPORT`, and sets the current user to the implicit `local` user for every test (so direct
`library.py`/`images.py`/`sessions.py` calls work with no HTTP request in the picture). Provider tests assert
exact request shapes against a mocked transport. Agent robustness tests use a `ScriptedBrain` double. The Google
auth flow is tested the same way (`base.TRANSPORT` mocked for the token exchange + userinfo calls) — see
`_login_as()` in `test_agent_and_api.py`, which two different tests reuse to simulate separate users and assert
their data never crosses. When adding a provider, add a wire test in `test_provider_wire.py`.

## Known gaps / good next tasks

- **Never run against real provider APIs.** Wire formats are verified only against mocks. First job with real keys:
  smoke-test each adapter, and confirm current model ids and image parameters in each vendor's docs (model
  suggestions in `registry.py` are starting points and go stale).
- Character consistency across pages is prompt-driven; evaluate how well it holds with real providers and tune
  `prompts.py` and the critic's rubric.
- Print pipeline (export ZIP to print-ready PDF) is not built.
- Multi-user now exists (Google sign-in, per-user directories, opt-in sharing), but sharing is view-only — there's
  no "copy this shared character/style into my own library" yet, so a shared asset can be looked at and reused as
  inspiration but not remixed in someone else's book directly.
- Only Google is wired up. Apple/GitHub/email+password would each add a new `app/auth.py`-style module and a new
  `users.upsert_from_*()`; the per-user storage and session-cookie layers underneath don't change.
- No rate limiting beyond the per-turn image cap and no per-account request throttling (a signed-in user can still
  hammer `/api/sessions/{id}/chat`).
