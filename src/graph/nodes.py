import re
import time
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from src.graph.state import AgentState
from src.agents.planner import PLANNER_SYSTEM_PROMPT
from src.utils.logger import get_logger

logger = get_logger("nodes")

_RETRY_PATTERN = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)

# City keyword → canonical city name (used by extract_metadata)
_CITY_MAP = {
    "paris": "Paris",
    "london": "London",
    "tokyo": "Tokyo",
    "new york": "New York",
    "berlin": "Berlin",
}

# Lazy-initialised bound model (avoids circular imports at module load time)
_model = None


def _get_model():
    global _model
    if _model is None:
        from src.agents.base import get_model
        from src.tools import ALL_TOOLS
        _model = get_model(temperature=0, bind_tools=ALL_TOOLS)
    return _model


# ── Node 1: Metadata Extraction 

def extract_metadata(state: AgentState) -> dict:
    """
    Pre-processing node that runs before every agent call.

    - Resets tool_call_count to 0 (prevents carry-over between turns)
    - Detects the destination city from the latest user message
    - Detects a budget figure (e.g. "$1500") from the latest user message
    """
    messages = state.get("messages", [])
    updates: dict = {"tool_call_count": 0}

    if not messages:
        return updates

    last_content = getattr(messages[-1], "content", "").lower()

    for keyword, city in _CITY_MAP.items():
        if keyword in last_content:
            updates["current_city"] = city
            logger.info(f"Detected city: {city}")
            break

    budget_match = re.search(r"\$(\d[\d,]*(?:\.\d+)?)", last_content)
    if budget_match:
        budget = float(budget_match.group(1).replace(",", ""))
        updates["total_budget"] = budget
        logger.info(f"Detected budget: ${budget}")

    return updates


#  Node 2: Agent (LLM call) 

def call_model(state: AgentState) -> dict:
    """
    Core agent node — sends the conversation history to the LLM and returns
    either a tool-call request or a final human-readable answer.
    Increments tool_call_count whenever a tool is requested.

    Automatically retries on 429 rate-limit errors using the delay
    reported by the API itself (free tier: 5 req/min on gemini-2.5-flash).
    """
    messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT)] + state["messages"]

    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            response = _get_model().invoke(messages)
            break
        except Exception as e:
            err = str(e)
            if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < max_retries - 1:
                match = _RETRY_PATTERN.search(err)
                wait = int(float(match.group(1))) + 3 if match else 30
                logger.warning(f"Rate limited — waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                print(f"\n  [rate limited — retrying in {wait}s...]", flush=True)
                time.sleep(wait)
            else:
                raise

    count = state.get("tool_call_count", 0)
    if hasattr(response, "tool_calls") and response.tool_calls:
        count += 1
        names = [tc["name"] for tc in response.tool_calls]
        logger.info(f"Tool call #{count}: {names}")

    return {"messages": [response], "tool_call_count": count}


# ── Node 3: Circuit Breaker 

def circuit_breaker(state: AgentState) -> dict:
    """
    Safety node — fires when the agent exceeds MAX_TOOL_CALLS or enters a
    repetitive loop. Injects a graceful error message and terminates the run.
    """
    count = state.get("tool_call_count", 0)
    logger.warning(f"Circuit breaker triggered after {count} tool calls.")

    msg = AIMessage(
        content=(
            "I've reached my processing limit for this request. "
            "This usually means the destination or route isn't in my database, "
            "or the question is ambiguous. Please try rephrasing, or ask about "
            "a supported city (Paris, London, Tokyo, New York, Berlin)."
        )
    )
    return {"messages": [msg]}


# ── Node 4: Researcher ───────────────────────────────────────────────────────

def researcher_node(state: AgentState) -> dict:
    """
    Lightweight data-lookup node — handles simple factual queries
    (e.g. "what hotels are in Tokyo?") by calling DB tools directly.
    Routes here when route_intent detects a research-only question.
    Bypasses the full planning workflow to avoid unnecessary LLM calls.
    """
    from src.tools.db_tools import fetch_hotels, fetch_flights, fetch_activities
    import json

    last_content = getattr(state["messages"][-1], "content", "").lower()
    city = state.get("current_city", "")

    if "hotel" in last_content and city:
        result = fetch_hotels.invoke({"city": city})
        label = f"Hotels in {city}"
    elif "flight" in last_content and city:
        result = fetch_flights.invoke({"origin": "TLV", "destination": city})
        label = f"Flights from TLV to {city}"
    elif "activit" in last_content and city:
        result = fetch_activities.invoke({"city": city})
        label = f"Activities in {city}"
    else:
        result = "Please specify hotels, flights, or activities along with a city name."
        label = "Research"

    content = f"**{label}:**\n{result}"
    logger.info("Researcher node completed query for city=%s.", city or "unknown")
    return {"messages": [AIMessage(content=content)]}


# ── Node 5: Reviewer ─────────────────────────────────────────────────────────

def reviewer_node(state: AgentState) -> dict:
    """
    Quality-control node — automatically critiques Marco's final travel plan.
    Only runs when the agent produced a full plan (city detected + 3+ tool calls).
    Appends a structured review to the conversation as a follow-up AIMessage.
    """
    from src.agents.reviewer import review_plan
    last_msg = state["messages"][-1]
    content = last_msg.content
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    review = review_plan(str(content))
    logger.info("Reviewer node completed critique.")
    return {"messages": [AIMessage(content=f"\n---\n**Plan Review (auto):**\n{review}")]}


# ── Tool Node (prebuilt) ──────────────────────────────────────────────────────

def build_tools_node():
    from src.tools import ALL_TOOLS
    return ToolNode(ALL_TOOLS)
