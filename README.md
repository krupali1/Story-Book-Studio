# Storybook Studio

A deployable AI design agent for illustrated children's picture books. You chat with it; it proposes
directions, designs characters, generates and critiques illustrations, and keeps a library of characters,
styles, palettes and finished books. Every approval and rejection is logged, and the agent rewrites a
**taste profile** after each book, so the next one starts smarter.

You choose the AI behind each job, from the Providers button in the top bar:

| Job | What it does | Providers |
|---|---|---|
| **Brain** | Conversation, story, prompt writing, drives the tools | Anthropic (Claude), OpenAI, Google Gemini, any OpenAI-compatible endpoint |
| **Critic** | Looks at each generated image and reviews it (needs a vision model) | Same list as Brain (defaults to the Brain's provider) |
| **Illustrator** | Generates the pictures | OpenAI Images, Google Gemini Images, Stability AI |

With no API keys the app starts in **demo mode** (scripted agent, placeholder art) so you can click through everything first.

## Run it

```bash
cp .env.example .env        # add at least one API key and a STUDIO_PASSWORD
docker compose up --build   # http://localhost:8000
```

Without Docker:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...     # whichever you use
uvicorn app.main:app --port 8000
```

## Deploy

Any host that runs a container works (Render, Railway, Fly.io, a VM with Docker). Three things matter:

1. **Persistent volume mounted at `/data`.** Your library, sessions, images and taste profile live there. Without a volume they vanish on redeploy.
2. **Set `STUDIO_PASSWORD`.** Otherwise anyone with the URL can spend your API credits. Serve over HTTPS (your host normally does this).
3. **Keys as environment variables** in the host's dashboard. Keys are never sent to the browser or written to disk.

`render.yaml` is a ready blueprint for Render (persistent disks need a paid instance). Run **one** worker/instance: storage is plain files.

## Choosing providers

Open **Providers** in the top bar and pick a provider and model per job. The choice is saved on the server.
Only providers whose key is set are selectable. You can type any model id, so new releases work without a code change;
the suggestions in the dropdown are only starting points, so check each vendor's docs for current ids.

| Provider | Env var(s) | Notes |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | default model `claude-sonnet-5` |
| OpenAI | `OPENAI_API_KEY` | chat + `gpt-image-1` images; images accept reference pictures |
| Google Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | chat + image model; images accept reference pictures |
| Stability AI | `STABILITY_API_KEY` | images only; **no reference images**, so character consistency relies on prompt text |
| OpenAI-compatible | `COMPAT_BASE_URL`, `COMPAT_API_KEY`, `COMPAT_MODEL` | OpenRouter, Ollama, Groq, vLLM, LM Studio... pick a vision model to use it as Critic |

Starting defaults can also come from `BRAIN_PROVIDER`, `BRAIN_MODEL`, `CRITIC_*`, `IMAGE_*`. Precedence: Providers dialog, then env vars, then the first provider with a key, then demo.

A good starting mix: Claude or Gemini as Brain and Critic, and whichever image model gives you the look you want. Reference-image support
(OpenAI, Gemini) is the main lever for keeping a character consistent across pages.

## How it works

The Brain runs an agent loop. Instead of vendor-specific function calling, tools use a plain-text JSON protocol in the prompt,
so **every provider behaves the same**, including local models. Tools: `search_library`, `read_asset`, `generate_image`,
`critique_image`, `extract_palette`, `make_moodboard`, `save_asset`, `log_feedback`, `update_taste`.

The prompt encodes the workflow with approval gates: brief, directions, character sheet, style lock, key pages, remaining pages, save and retro.
It never spends a big image batch before you approve a direction and a character.

What accumulates on disk (`DATA_DIR`):

```
library/characters|styles|palettes|books/<slug>/asset.json   versioned assets (image ids inside)
images/img_xxxxxxxx.png (+ .json with the exact prompt, provider, model)
sessions/<id>.json     one chat per book
taste.md               distilled preferences, rewritten by the agent, editable in the Taste tab
taste_history/         previous versions of taste.md
feedback.jsonl         every keep/reject with tags and reasons
settings.json          provider choices
```

The agent reads the taste profile, a library digest and a feedback summary at the start of every turn. **Explore mode** tells it to
deliberately step outside your taste profile.

**Cost guard rails:** `MAX_IMAGES_PER_TURN` (default 6) caps image generations per message; `MAX_AGENT_STEPS` caps tool rounds.

## Handing off to print

A finished book saved by the agent can be downloaded from the Library as a ZIP (`book.json`, readable `story.md`, page images).
That is the seam for a later print pipeline (typesetting, bleed, PDF/X); it isn't built yet.

## Extending

- **New LLM provider:** add a class with `async complete(system, messages, max_tokens) -> str` in `app/providers/llm.py`, register it in `app/registry.py`.
- **New image provider:** a class with `supports_reference` and `async generate(prompt, negative, aspect, references) -> bytes` in `app/providers/image.py`, then register it.
- **New tool:** decorate an async function with `@tool(...)` in `app/tools.py`; it appears in the agent's prompt automatically.

## Tests

```bash
pip install -r requirements-dev.txt && pytest -q
```

35 tests cover the library, tools, agent loop (malformed tool blocks, call caps, step limit, provider failures), the HTTP API and
auth, and each provider adapter's request and response shape against mocked endpoints.

## Limits worth knowing

- **Live provider calls are not covered by the tests.** Adapters follow each vendor's documented REST API and were checked against mocked
  responses (plus a real 401 from Anthropic to confirm the request reaches the API). Do one real run per provider after deploying; if a vendor
  changed a field or model id, the error is shown in the chat and the fix is in one adapter.
- Single user, single instance, file storage. Fine for a personal studio; a team would want accounts and a database.
- Character consistency is best-effort. Reference images plus restating the character description in every prompt help a lot, but no image model is perfect.
- Replies arrive one step at a time (text, then tool results), not token by token.
