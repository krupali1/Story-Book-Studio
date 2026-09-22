# Storybook Studio: project context for Claude Code

A self-hosted AI design agent for illustrated children's picture books. The user chats with it; it plans,
writes prompts, generates and critiques illustrations, and keeps a persistent library plus a taste profile
that improves across books. Personal project (not client work). Owner: Ankit.

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q                    # 35 tests, no network or API keys needed
uvicorn app.main:app --reload --port 8000
docker compose up --build              # needs .env (copy .env.example)
```

With no API keys the app runs in **demo mode** (MockLLM + MockImage), so the UI is fully clickable offline.

## Architecture in one paragraph

FastAPI backend + one vanilla-JS file (`static/index.html`, no build step). Three independently selectable AI
**roles**: `brain` (chat/planning/prompts, drives tools), `critic` (vision review of generated images),
`image` (illustration generation). Provider+model per role is chosen in the Providers dialog and saved to
`DATA_DIR/settings.json`. API keys come **only** from server env vars and are never sent to the browser
(`registry.describe()` exposes env var *names*, never values). Storage is plain JSON/Markdown/PNG files under
`DATA_DIR`; there is no database. Because of that, run exactly **one** worker.

## Code map

- `app/config.py`: env config, `load_settings/save_settings`, `ROLES`.
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
- `app/library.py`: assets (character/style/palette/book), versioned taste profile, JSONL feedback log.
- `app/images.py`: image store (`img_XXXXXXXX` ids, path-traversal safe), vision downscale, palette, moodboard.
- `app/main.py`: routes, optional password auth (`STUDIO_PASSWORD`, HMAC cookie), SSE chat endpoint that runs the
  turn as a detached task so it survives client disconnects, and `/api/books/{slug}/export`.

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

`tests/conftest.py` isolates `DATA_DIR`, clears all provider env vars, resets `base.TRANSPORT` and disables auth
for every test. Provider tests assert exact request shapes against a mocked transport. Agent robustness tests use
a `ScriptedBrain` double. When adding a provider, add a wire test in `test_provider_wire.py`.

## Known gaps / good next tasks

- **Never run against real provider APIs.** Wire formats are verified only against mocks. First job with real keys:
  smoke-test each adapter, and confirm current model ids and image parameters in each vendor's docs (model
  suggestions in `registry.py` are starting points and go stale).
- Character consistency across pages is prompt-driven; evaluate how well it holds with real providers and tune
  `prompts.py` and the critic's rubric.
- Print pipeline (export ZIP to print-ready PDF) is not built.
- Single-user by design. Multi-user would need a real DB, per-user data dirs and per-user auth.
- No rate limiting beyond the per-turn image cap and login delay.
