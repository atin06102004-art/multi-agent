import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq


from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM - original model kept
# =========================
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    # Groq's free/on-demand tier caps this model at 8000 tokens/min, and
    # that cap covers prompt + completion together. Bounding completion
    # size here leaves more of that budget for the (already-clipped)
    # prompt content.
    max_tokens=900,
)

# =========================
# State - original fields kept, new control fields added
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget state
    budget_results: str
    final_response: str

    # Human-in-the-loop state
    hitl_approved: bool
    hitl_feedback: str
    revision_count: int

    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


def _clip(text: str, max_chars: int = 900) -> str:
    """Cap a block of text going into a prompt so a chain of specialist
    results (flight/hotel/weather/budget/itinerary) can't blow past Groq's
    per-minute token limit when they're all concatenated into one prompt.
    """
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n…(truncated)"


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent - original behavior kept
# =========================
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")
    query = state["user_query"]

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports"))
        airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:900],
            airline_data=str(airlines)[:900],
        )

        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel flight planner."),
                HumanMessage(content=prompt),
            ]
        )
        flight_data = response.content
    except Exception as exc:
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent - original behavior kept
# =========================
def hotel_agent(state: TravelState):
    query = (
        f"Best hotels for "
        f"{state['user_query']}"
    )

    try:
        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )

    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }


# =========================
# Weather Agent - original behavior kept
# =========================
def weather_agent(state: TravelState):
    city = extract_destination(
        state["user_query"]
    )

    try:
        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""

    except Exception as exc:
        print(
            f"WEATHER AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Give general "
            "seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
        # extract_destination() + the weather/forecast lookups involve one
        # LLM call (destination extraction); keep the running total accurate.
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Budget Agent - new specialist
# =========================
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{_clip(state.get('flight_results', ''))}

Hotel Results:
{_clip(state.get('hotel_results', ''))}

Weather Results:
{_clip(state.get('weather_results', ''))}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent - original behavior extended with selected results
# =========================
def itinerary_agent(state: TravelState):
    revision_feedback = state.get("hitl_feedback", "")
    revision_note = (
        f"""
The user reviewed the previous draft and requested changes. Revise the
itinerary to address this feedback directly, keeping everything else that
still applies:
{revision_feedback}
"""
        if revision_feedback
        else ""
    )

    prompt = f"""
Create a complete travel itinerary using the research below. This draft is
shown directly to the user for approval, so it must surface the actual
findings from each specialist agent, not generic placeholder advice.
{revision_note}
User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results (from the flight agent):
{_clip(state.get('flight_results', '')) or 'Not researched for this request.'}

Hotel Results (from the hotel agent):
{_clip(state.get('hotel_results', '')) or 'Not researched for this request.'}

Weather Results (from the weather agent, includes current temperature and forecast):
{_clip(state.get('weather_results', '')) or 'Not researched for this request.'}

Budget Results (from the budget agent):
{_clip(state.get('budget_results', '')) or 'Not researched for this request.'}

Structure the draft with these sections, in this order:
1. Trip Overview — destination, dates/duration, origin if known
2. Flights — one bullet per airline from the Flight Results above, each
   bullet pairing that airline with its own specific fare range (e.g.
   "Delta Air Lines — $650–900 round trip"). Never give one combined
   range for all airlines. Do not invent generic "book flights in
   advance" filler if real data was returned.
3. Hotels — one bullet per hotel/option from the Hotel Results above,
   each bullet pairing that hotel's name directly with its own price
   range (e.g. "Remm Akihabara Hotel — ¥8,000–12,000/night"). Never
   list hotel names and prices in separate groups.
4. Weather — a detailed bullet list, broken out per city if the data
   covers more than one city. For each city include separate bullets
   for: current temperature, feels-like temperature, humidity,
   condition, and forecast range. Do not compress this into a single
   paragraph.
5. Day-by-Day Plan — activities per day, referencing the specific hotel
   and weather info where relevant (e.g. pack for the actual forecast)
6. Budget Snapshot — pull figures from the Budget Results above

If a section's "Results" say the data was unavailable or unresearched,
say so plainly in that section instead of fabricating numbers.

Make the itinerary practical and easy to follow. This is a clear draft
ready for human review.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "itinerary": response.content,
        # Clear any prior feedback/decision now that a fresh draft exists,
        # and mark it as not-yet-approved so routing sends it to the human
        # approval node instead of straight through to final_agent.
        "hitl_approved": False,
        "hitl_feedback": "",
        "messages": [AIMessage(content="Draft itinerary created.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop: approval gate
# =========================
def human_approval_agent(state: TravelState):
    """Pause the graph and hand the draft itinerary to the user.

    Resuming happens via ``Command(resume={"approved": bool, "feedback": str})``
    passed to ``travel_graph.invoke``/``.stream`` with the same thread_id. See
    ``resume_travel_agent`` below.
    """
    decision = interrupt(
        {
            "type": "itinerary_approval",
            "question": (
                "Here's the draft itinerary. Approve it, or describe what "
                "should change."
            ),
            "itinerary": state.get("itinerary", ""),
        }
    )

    approved = bool(decision.get("approved", False)) if isinstance(decision, dict) else False
    feedback = (
        str(decision.get("feedback", "")).strip()
        if isinstance(decision, dict)
        else ""
    )

    summary = (
        "User approved the draft itinerary."
        if approved
        else f"User requested revisions: {feedback or '(no details given)'}"
    )

    return {
        "hitl_approved": approved,
        "hitl_feedback": feedback,
        "revision_count": state.get("revision_count", 0) + (0 if approved else 1),
        "messages": [AIMessage(content=summary)],
    }


# =========================
# Final Response Agent - original format kept
# =========================
def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights (one bullet per airline, each paired with its own fare range — never one combined range for all airlines):
{_clip(state.get('flight_results', '')) or 'Not researched for this request.'}

Hotels (one bullet per hotel, each hotel name paired directly with its own price range — never listed separately from prices):
{_clip(state.get('hotel_results', '')) or 'Not researched for this request.'}

Weather (detailed bullets, broken out per city if more than one city — separate bullets for current temp, feels-like, humidity, condition, forecast range):
{_clip(state.get('weather_results', '')) or 'Not researched for this request.'}

Budget Analysis (use the figures below):
{_clip(state.get('budget_results', '')) or 'Not researched for this request.'}

Draft Itinerary:
{_clip(state.get('itinerary', ''), max_chars=1600)}

Format the final answer as clean Markdown, following this exact structure
(this renders directly in the app, so the Markdown syntax matters):

## Trip Summary
---
A short paragraph: destination, duration, origin, and overall approach
given the budget.

## Flight Information
---
One bullet per airline, each bolded and paired with its own fare range,
e.g. `- **Delta Air Lines:** $650–900 round trip`. Note any layovers or
routing needed. If no live pricing was available, say so plainly here.

## Hotel Suggestions
---
One bullet per hotel/area, bolded name followed by its price range, e.g.
`- **Tokyo:** Remm Akihabara Hotel — ¥8,000–12,000/night`.

## Weather Information
---
Bullets per city covering current temperature, feels-like, humidity,
condition, and forecast range.

## Day-by-Day Itinerary
---
A `### Day N: <short title>` subheading for each day, followed by 2-4
bullet points of activities for that day. Reference the actual hotel and
weather info where relevant (e.g. pack for the real forecast).

## Estimated Budget
---
One bolded bullet per cost category pulled from the Budget Analysis
above, each with its own range, e.g.:
- **Flights:** ~$X (range)
- **Accommodation:** $X–Y for the entire stay
- **Transport:** $X–Y
- **Food and Sightseeing:** $X–Y

Followed by a `**Total:** Approximately $X to $Y` line summing the
ranges above.

## Final Recommendations
---
4-6 concise, actionable bullet points (booking timing, transit passes,
flexibility tips, local advice) — not a repeat of earlier sections.

Important:
- Use real Markdown syntax exactly as shown above (## headings, the
  literal `---` divider line right under each heading, **bold** labels,
  `-` bullets, `###` day subheadings) so it renders with real section
  headers and dividers instead of flat paragraphs.
- Pull real data from the sections above into each corresponding part
  below — don't fall back on generic advice ("book in advance", "check
  the weather") when actual data was returned.
- If a section's data above says it was unavailable, say so plainly
  instead of fabricating numbers.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Keep the response useful for real travel planning.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# Cap revision loops so a stuck/ambiguous feedback cycle can't loop forever.
MAX_REVISIONS = 3


def route_after_approval(state: TravelState) -> str:
    if state.get("hitl_approved", False):
        return "final_agent"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        # Out of revision attempts - finalize with whatever draft exists
        # rather than looping indefinitely.
        return "final_agent"
    return "itinerary_agent"


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_conditional_edges(
    "human_approval",
    route_after_approval,
    {"final_agent": "final_agent", "itinerary_agent": "itinerary_agent"},
)
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer - original persistence kept
# =========================
DATABASE_URL = get_database_url()
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    # When the graph is paused at the human_approval node, invoke()/ainvoke()
    # returns the state so far plus an "__interrupt__" key holding the
    # Interrupt object(s) raised by interrupt(). Surface that as a distinct
    # "awaiting approval" response instead of treating it as a final answer.
    interrupts = result.get("__interrupt__") or []
    if interrupts:
        payload = interrupts[0].value or {}
        draft_itinerary = payload.get("itinerary") or result.get("itinerary", "")
        question = payload.get(
            "question",
            "Here's the draft itinerary. Approve it, or describe what should change.",
        )

        return {
            "thread_id": thread_id,
            "requires_approval": True,
            "answer": question,
            "flight_results": result.get("flight_results", ""),
            "hotel_results": result.get("hotel_results", ""),
            "weather_results": result.get("weather_results", ""),
            "budget_results": result.get("budget_results", ""),
            "itinerary": draft_itinerary,
            "selected_agents": result.get("selected_agents", []),
            "trip_constraints": result.get("trip_constraints", {}),
            "supervisor_reasoning": result.get("supervisor_reasoning", ""),
            "guardrail_allowed": result.get("guardrail_allowed", True),
            "guardrail_reason": result.get("guardrail_reason", ""),
            "llm_calls": result.get("llm_calls", 0),
        }

    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message

    return {
        "thread_id": thread_id,
        "requires_approval": False,
        "answer": answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": result.get("itinerary", ""),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Run a travel-planning request through the full agent graph.

    Stops (returns with requires_approval=True) once the graph reaches the
    human_approval node. Call ``resume_travel_agent`` with the same
    thread_id to continue.
    """
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "final_response": "",
            "hitl_approved": False,
            "hitl_feedback": "",
            "revision_count": 0,
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume a graph paused at the human_approval interrupt.

    ``approved=True`` sends the draft straight to final_agent.
    ``approved=False`` sends ``feedback`` back to itinerary_agent for a
    revised draft, which pauses again at human_approval.
    """
    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        Command(resume={"approved": approved, "feedback": feedback}),
        config=config,
    )

    return _serialize_result(result, thread_id)
