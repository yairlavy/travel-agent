from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Central shared memory across all graph nodes.

    messages          — full conversation history (append-only via add_messages reducer)
    current_city      — destination extracted from the latest user message
    total_budget      — budget extracted from the latest user message (USD)
    tool_call_count   — incremented each time the agent requests a tool; guards against infinite loops
    validation_status — set by the validator node: "approved" | "blocked_injection"
                        | "blocked_scope" | "blocked_city" | "" (not yet validated)
    """

    messages: Annotated[list, add_messages]
    current_city: str
    total_budget: float
    tool_call_count: int
    validation_status: str
