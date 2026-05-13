from langchain_core.messages import HumanMessage
from langgraph.graph import END
from src.graph.state import AgentState
from src.utils.validators import detect_repetition


def _current_turn_messages(messages: list) -> list:
    """
    Return only the messages belonging to the current turn.
    Slices from the last HumanMessage onwards so detect_repetition
    doesn't false-positive on tool calls from previous sessions.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages

MAX_TOOL_CALLS = 8

# Keywords that signal the user wants raw data, not a full trip plan
_RESEARCH_TRIGGERS = {
    "what flights", "what hotels", "list hotels", "list flights",
    "show flights", "show hotels", "available flights", "available hotels",
    "what activities", "list activities", "show activities",
}

# Phrases that mean the user is asking what the agent remembers
_RECALL_TRIGGERS = {
    "what do i prefer", "what's my preference", "what are my preferences",
    "do you remember", "what airline do i", "my preference", "what food",
    "remember my", "what did i tell you", "my saved", "my profile",
    "what do you know about me", "my details",
}

# Keywords that signal the user is stating a preference to save
_PREFERENCE_KEYWORDS = {
    "i prefer", "i like", "my favourite", "my favorite",
    "always fly", "always travel with", "i'm traveling with",
    "traveling with", "travelling with", "people traveling",
    "kosher", "vegan", "vegetarian", "halal", "gluten",
}


def route_after_metadata(state: AgentState) -> str:
    """
    Conditional edge after extract_metadata — runs on every user message.

    Decision tree:
      1. User is asking what the agent remembers  → recall
      2. User is stating a preference to save     → update_preferences
      3. Everything else                          → validator (normal flow)
    """
    last_content = getattr(state["messages"][-1], "content", "").lower()

    for trigger in _RECALL_TRIGGERS:
        if trigger in last_content:
            return "recall"

    for kw in _PREFERENCE_KEYWORDS:
        if kw in last_content:
            return "update_preferences"

    return "validator"


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


def route_after_validator(state: AgentState) -> str:
    """
    Conditional edge after the validator node.

    If blocked → END immediately (rejection message already in State).
    If approved → check whether it's a research-only query or a full plan request.
    """
    status = state.get("validation_status", "approved")
    if status != "approved":
        return END

    last_content = getattr(state["messages"][-1], "content", "").lower()
    for trigger in _RESEARCH_TRIGGERS:
        if trigger in last_content:
            return "researcher"
    return "agent"


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function — decides the next node after the agent runs.

    Decision tree:
      1. tool_call_count >= MAX_TOOL_CALLS       → circuit_breaker
      2. Repetitive identical tool call detected  → circuit_breaker
      3. Last message contains tool_calls         → tools (keep working)
      4. Full plan (5+ tools + city) + is_admin   → reviewer → summarizer
      5. Final answer (any session)               → summarizer
    """
    last = state["messages"][-1]
    count = state.get("tool_call_count", 0)
    is_admin = state.get("is_admin", False)

    if count >= MAX_TOOL_CALLS:
        return "circuit_breaker"

    if detect_repetition(_current_turn_messages(state["messages"])):
        return "circuit_breaker"

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # Reviewer only runs for ADMIN sessions with a full plan
    if is_admin and count >= 5 and state.get("current_city"):
        return "reviewer"

    # All final answers (admin simple + non-admin) go through summarizer
    return "summarizer"
