"""
Streamlit frontend for the Multi-Agent Travel Planner
(LangGraph + MCP + Supervisor + Guardrails).

This talks directly to backend.py -- it does NOT go through the FastAPI
app.py / REST endpoints. Streamlit is both the UI and the process that
drives the LangGraph graph, so `run_travel_agent` is called as a plain
Python function.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st

from backend import run_travel_agent, resume_travel_agent


# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="TripMate AI — Multi-Agent Travel Planner",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ TripMate AI")
st.caption(
    "Multi-agent travel planner — Let's you plan your Trip ."
)


# =========================
# Session state
# =========================
def _reset_conversation():
    st.session_state.thread_id = None
    st.session_state.chat_history = []          # list[{"role": "user"/"assistant", "content": str}]
    st.session_state.last_result = None          # most recent full result dict, for the sidebar
    st.session_state.awaiting_approval = False   # True once the graph pauses at human_approval
    st.session_state.pending_itinerary = ""      # draft shown while awaiting_approval is True


if "thread_id" not in st.session_state:
    _reset_conversation()


# =========================
# Sidebar — conversation + agent internals
# =========================
with st.sidebar:
    st.subheader("Conversation")
    st.text(f"Thread ID: {st.session_state.thread_id or '(new)'}")

    if st.button("🔄 Start a new trip", use_container_width=True):
        _reset_conversation()
        st.rerun()

    result = st.session_state.last_result
    if result:
        st.divider()
        st.subheader("Agent internals")

        st.markdown(f"**LLM calls so far:** {result.get('llm_calls', 0)}")

        if not result.get("guardrail_allowed", True):
            st.error(f"Guardrail blocked: {result.get('guardrail_reason', '')}")

        selected = result.get("selected_agents") or []
        if selected:
            st.markdown("**Supervisor selected agents:**")
            st.write(", ".join(selected))

        reasoning = result.get("supervisor_reasoning")
        if reasoning:
            with st.expander("Supervisor reasoning"):
                st.write(reasoning)

        constraints = result.get("trip_constraints")
        if constraints and any(constraints.values()):
            with st.expander("Extracted trip constraints"):
                st.json(constraints)

        for label, key in [
            ("✈️ Flight results", "flight_results"),
            ("🏨 Hotel results", "hotel_results"),
            ("🌤 Weather results", "weather_results"),
            ("💰 Budget analysis", "budget_results"),
            ("📝 Draft itinerary", "itinerary"),
        ]:
            value = result.get(key)
            if value:
                with st.expander(label):
                    st.write(value)


# =========================
# Render chat history
# =========================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# Chat input — start / continue a planning request
# =========================
user_input = st.chat_input(
    "e.g. Plan a 3-day trip to Tokyo with a budget of $1200",
    disabled=st.session_state.awaiting_approval,
)

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Planning..."):
            try:
                result = run_travel_agent(
                    user_input=user_input,
                    thread_id=st.session_state.thread_id,
                )
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.stop()

        st.session_state.thread_id = result["thread_id"]
        st.session_state.last_result = result

        if result.get("requires_approval"):
            st.session_state.awaiting_approval = True
            st.session_state.pending_itinerary = result.get("itinerary", "")
            st.markdown(result["answer"])
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["answer"]}
            )
        else:
            st.markdown(result["answer"])
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["answer"]}
            )

    st.rerun()


# =========================
# Human-in-the-loop — approve or request changes to the draft itinerary
# =========================
if st.session_state.awaiting_approval:
    with st.chat_message("assistant"):
        st.markdown("**Draft itinerary — review below and approve or request changes.**")
        with st.expander("📝 Draft itinerary", expanded=True):
            st.markdown(st.session_state.pending_itinerary or "_No draft available._")

        approve_col, revise_col = st.columns(2)
        approve_clicked = approve_col.button(
            "✅ Approve", use_container_width=True, key="approve_btn"
        )

        with revise_col.popover("✏️ Request changes", use_container_width=True):
            feedback = st.text_area(
                "What should change?",
                key="revision_feedback_input",
                placeholder="e.g. Swap the hotel for something cheaper, add a museum day",
            )
            revise_clicked = st.button("Submit revision", key="submit_revision_btn")

        if approve_clicked:
            with st.spinner("Finalizing your trip..."):
                try:
                    result = resume_travel_agent(
                        thread_id=st.session_state.thread_id,
                        approved=True,
                    )
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")
                    st.stop()

            st.session_state.last_result = result
            st.session_state.awaiting_approval = result.get("requires_approval", False)
            st.session_state.pending_itinerary = result.get("itinerary", "")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["answer"]}
            )
            st.rerun()

        if revise_clicked:
            if not feedback.strip():
                st.warning("Add a note on what should change before submitting.")
            else:
                with st.spinner("Revising the itinerary..."):
                    try:
                        result = resume_travel_agent(
                            thread_id=st.session_state.thread_id,
                            approved=False,
                            feedback=feedback.strip(),
                        )
                    except Exception as exc:
                        st.error(f"Something went wrong: {exc}")
                        st.stop()

                st.session_state.last_result = result
                st.session_state.awaiting_approval = result.get(
                    "requires_approval", True
                )
                st.session_state.pending_itinerary = result.get("itinerary", "")
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Revising the itinerary based on your feedback: "
                            f"{feedback.strip()}"
                        ),
                    }
                )
                if not st.session_state.awaiting_approval:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result["answer"]}
                    )
                st.rerun()
