"""
FastAPI backend for TripMate AI.

This is a pure JSON REST API — there is no server-rendered HTML, no
templates, and no static assets here. The interactive UI lives entirely
in streamlit_app.py, which talks to backend.py directly. This app.py
exists so the same LangGraph agent graph can also be driven over HTTP
(e.g. from another service, a mobile client, or curl/Postman) without
requiring Streamlit.
"""

from contextlib import asynccontextmanager
import traceback

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend import run_travel_agent

# Kept from the original project so the synchronous LangGraph agent
# functions can call async MCP helpers while running inside FastAPI's
# own event loop.
import nest_asyncio

nest_asyncio.apply()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # backend.py opens its PostgreSQL checkpointer connection at import
    # time, so importing it here (via run_travel_agent) fails fast with a
    # clear error if DATABASE_URL / the other required env vars are
    # missing, instead of failing on the first request.
    yield


app = FastAPI(
    title="TripMate AI API",
    description=(
        "LangGraph multi-agent travel planner — Supervisor routing, an "
        "input guardrail, and MCP tool calls (Tavily, AviationStack, "
        "OpenWeather). JSON only, no server-rendered frontend."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Permissive by default so any client (Streamlit on a different port, a
# separate frontend, curl) can call the API during development. Tighten
# allow_origins to your real frontend's origin(s) before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


@app.get("/")
async def root():
    return {
        "service": "TripMate AI API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/api/travel"],
    }


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                },
            )

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        print("ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "TripMate AI API is running",
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "mcp_tools",
        ],
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )