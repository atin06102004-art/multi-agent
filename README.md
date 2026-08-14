# TripMate AI — Multi-Agent Travel Planner (LangGraph + MCP + Supervisor + Guardrails + HITL)

A multi-agent travel-planning system built with LangGraph and MCP. A
Supervisor agent routes each request to the specialist agents it actually
needs, an input guardrail filters out non-travel/harmful requests, and a
Human-in-the-Loop (HITL) step pauses the graph so the user can approve or
request revisions to the draft itinerary before it's finalized.

There is **no HTML/CSS/JS frontend** in this project. The UI is
[Streamlit](https://streamlit.io/), and FastAPI is exposed purely as a
JSON REST API for anything that needs to drive the same agent graph over
HTTP.

## Architecture

- **`backend.py`** — the LangGraph graph itself: state definition,
  supervisor + guardrail, the flight/hotel/weather/budget/itinerary
  agents, the `interrupt()`-based human-approval node, the final response
  agent, and a PostgreSQL checkpointer so conversations survive restarts.
- **`mcp_client.py`** — connects to three MCP servers: Tavily (hotel/web
  search), AviationStack (via `uvx`, flight/airport data), and a local
  weather server.
- **`custom_weather_mcp_server.py`** — a small FastMCP server wrapping the
  OpenWeather API, run as a stdio subprocess by `mcp_client.py`.
- **`streamlit_app.py`** — the interactive UI. Calls `backend.py`
  directly (no HTTP hop), and includes a sidebar showing supervisor
  routing, guardrail status, and each specialist agent's output, plus an
  in-chat approve/revise flow for the HITL step. **This is the primary
  way to use the app.**
- **`app.py`** — a FastAPI app exposing the same graph as two JSON
  endpoints, for any non-Streamlit client (another service, a mobile
  app, curl/Postman). No templates, no static assets, no rendered pages
  — every response is JSON.

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) installed and on `PATH` (needed for
  `uvx`, which runs the AviationStack MCP server)
- A PostgreSQL database reachable from wherever you run this (e.g. a free
  Render/Neon/Supabase instance) — used only for LangGraph checkpointing
- API keys: Groq, Tavily, AviationStack, OpenWeather

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\Activate.ps1 on Windows

pip install -r requirements.txt

cp .env.example .env
# then fill in DATABASE_URL, GROQ_API_KEY, TAVILY_API_KEY,
# AVIATIONSTACK_API_KEY, OPENWEATHER_API_KEY
```

## Run it

**Streamlit (primary UI):**

```bash
streamlit run streamlit_app.py
```

Opens at http://localhost:8501.

**FastAPI (JSON API only):**

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Interactive API docs at http://127.0.0.1:8000/docs.

**Standalone weather MCP server** (only needed if you want to test it in
isolation — `mcp_client.py` already launches it automatically as a
subprocess):

```bash
python custom_weather_mcp_server.py
```

## Docker

```bash
docker build -t tripmate-ai .
docker run --env-file .env -p 8501:8501 tripmate-ai        # Streamlit UI

docker run --env-file .env -p 8000:8000 tripmate-ai \
  uvicorn app:app --host 0.0.0.0 --port 8000                # JSON API instead
```

Or run both at once:

```bash
docker compose up --build
```

## API reference (`app.py`)

- `POST /api/travel` — start or continue a planning thread.
  `{ "message": "<user prompt>", "thread_id": "optional-existing-id" }`
  Returns `requires_approval: true` and a draft `itinerary` once the
  graph reaches the HITL step.
- `POST /api/travel/approve` — approve or request a revision for the
  paused draft.
  `{ "thread_id": "<id>", "approved": true|false, "feedback": "optional" }`
- `GET /health` — health check.

## Security note

Rotate every credential that was ever in a `.env` you shared or
committed (database password, Groq/Tavily/AviationStack/OpenWeather
keys) — treat any key that left your machine as compromised. Keep
secrets only in `.env` (already git- and docker-ignored), never in code
or in `requirements.txt`/README.

## Notes

- `backend.py` opens its PostgreSQL checkpointer connection when it's
  imported, so a missing/invalid `DATABASE_URL` (or the other required
  env vars) fails fast and loudly instead of on the first request.
- The input guardrail and supervisor routing both fail open (default to
  "allowed" / the full agent pipeline) if the LLM returns malformed JSON,
  so a transient model hiccup degrades gracefully instead of breaking the
  app.
- Tests are not included; exercise the graph via the Streamlit UI or the
  API endpoints directly.

## License

See `LICENSE`.
