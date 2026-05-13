from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Central shared memory across all graph nodes.

    messages              — full conversation history (append-only via add_messages reducer)
    current_city          — destination extracted from the latest user message
    total_budget          — budget extracted from the latest user message (USD)
    tool_call_count       — incremented each time the agent requests a tool; guards loops
    validation_status     — set by validator: "approved" | "blocked_injection" |
                            "blocked_scope" | "blocked_city" | "blocked_harm"
    preferred_airline     — user's preferred airline, persisted across sessions
    food_preference       — user's dietary preference (kosher, vegan …), persisted
    num_travelers         — number of people traveling, persisted across sessions
    is_admin              — True when session ID ends with ADMIN00; controls reviewer routing
    conversation_summary  — compact summary of past messages produced by summarizer_node
    """

    messages: Annotated[list, add_messages]
    current_city: str
    total_budget: float
    tool_call_count: int
    validation_status: str
    preferred_airline: str
    food_preference: str
    num_travelers: int
    is_admin: bool
    conversation_summary: str
    travel_preferences: str
