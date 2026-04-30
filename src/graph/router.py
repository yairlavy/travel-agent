from langgraph.graph import END
from src.graph.state import AgentState
from src.utils.validators import detect_repetition

MAX_TOOL_CALLS = 8


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function — decides the next node after the agent runs.

    Decision tree:
      1. tool_call_count >= MAX_TOOL_CALLS  → circuit_breaker  (hard cap)
      2. Repetitive identical tool call detected → circuit_breaker  (loop guard)
      3. Last message contains tool_calls     → tools           (keep working)
      4. Otherwise                            → END             (final answer ready)
    """
    last = state["messages"][-1]
    count = state.get("tool_call_count", 0)

    if count >= MAX_TOOL_CALLS:
        return "circuit_breaker"

    if detect_repetition(state["messages"]):
        return "circuit_breaker"

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return END
