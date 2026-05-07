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


# ── Tool Node (prebuilt) ──────────────────────────────────────────────────────

def build_tools_node():
    from src.tools import ALL_TOOLS
    return ToolNode(ALL_TOOLS)
