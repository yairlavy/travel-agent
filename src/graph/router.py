from langgraph.graph import END
from src.graph.state import AgentState
from src.utils.validators import detect_repetition

MAX_TOOL_CALLS = 8

# Keywords that signal the user wants raw data, not a full trip plan
_RESEARCH_TRIGGERS = {
    "what flights", "what hotels", "list hotels", "list flights",
    "show flights", "show hotels", "available flights", "available hotels",
    "what activities", "list activities", "show activities",
}


def route_intent(state: AgentState) -> str:
    """
    Conditional edge after extract_metadata.
    Routes simple data-lookup questions to the researcher node,
    and everything else (planning, general questions) to the main agent.
    """
    last_content = getattr(state["messages"][-1], "content", "").lower()
    for trigger in _RESEARCH_TRIGGERS:
        if trigger in last_content:
            return "researcher"
    return "agent"


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function — decides the next node after the agent runs.

    Decision tree:
      1. tool_call_count >= MAX_TOOL_CALLS  → circuit_breaker  (hard cap)
      2. Repetitive identical tool call detected → circuit_breaker  (loop guard)
      3. Last message contains tool_calls     → tools           (keep working)
      4. Final answer + city detected + 3+ tool calls → reviewer  (full plan)
      5. Otherwise                            → END             (simple answer)
    """
    last = state["messages"][-1]
    count = state.get("tool_call_count", 0)

    if count >= MAX_TOOL_CALLS:
        return "circuit_breaker"

    if detect_repetition(state["messages"]):
        return "circuit_breaker"

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    if count >= 3 and state.get("current_city"):
        return "reviewer"

    return END
