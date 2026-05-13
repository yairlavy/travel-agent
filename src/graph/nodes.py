import re
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from src.graph.state import AgentState
from src.agents.planner import PLANNER_SYSTEM_PROMPT
from src.utils.logger import get_logger

logger = get_logger("nodes")

# Prompt caching — base SystemMessage built once at module level.
# call_model reuses this instance when no user profile is present,
# avoiding string reconstruction on every LLM call.
_BASE_SYSTEM_MSG = SystemMessage(content=PLANNER_SYSTEM_PROMPT)

_RETRY_PATTERN = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)

# City keyword → canonical city name (used by extract_metadata)
_CITY_MAP = {
    "paris": "Paris",
    "london": "London",
    "tokyo": "Tokyo",
    "new york": "New York",
    "berlin": "Berlin",
}

# Known airline names for preference detection
_AIRLINE_NAMES = [
    "el al", "emirates", "lufthansa", "british airways", "air france",
    "united", "virgin atlantic", "ryanair", "turkish airlines", "klm",
    "swiss", "tap", "wizz", "easyjet",
]

# Dietary preferences for preference detection
_FOOD_PREFS = [
    "kosher", "vegan", "vegetarian", "halal", "gluten-free", "gluten free",
]

# Lazy-initialised bound model (avoids circular imports at module load time)
_model = None


def _get_model():
    global _model
    if _model is None:
        from src.agents.base import get_model
        from src.tools import ALL_TOOLS
        _model = get_model(temperature=0, bind_tools=ALL_TOOLS)
    return _model


# ── Node 1: Metadata Extraction ──────────────────────────────────────────────

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


# ── Node 2: Update Preferences ───────────────────────────────────────────────

def update_preferences(state: AgentState) -> dict:
    """
    Dedicated node for extracting and persisting user preferences.
    Detects preferred airline, dietary needs, and number of travelers
    from the latest message and saves them to State (checkpointed to disk).
    """
    last_content = getattr(state["messages"][-1], "content", "").lower()
    updates: dict = {}

    for airline in _AIRLINE_NAMES:
        if airline in last_content:
            updates["preferred_airline"] = airline.title()
            logger.info(f"Preference detected — airline: {airline.title()}")
            break

    for food in _FOOD_PREFS:
        if food in last_content:
            updates["food_preference"] = food
            logger.info(f"Preference detected — food: {food}")
            break

    traveler_match = re.search(
        r"(\d+)\s*(people|person|traveler|travellers|passenger|of us|pax)",
        last_content,
    )
    if traveler_match:
        updates["num_travelers"] = int(traveler_match.group(1))
        logger.info(f"Preference detected — travelers: {traveler_match.group(1)}")

    return updates


# ── Node 3: Recall ────────────────────────────────────────────────────────────

def recall_node(state: AgentState) -> dict:
    """
    Answers questions about saved preferences and history directly from State,
    without calling the LLM. Runs when the user asks what the agent remembers.
    """
    parts = []

    airline = state.get("preferred_airline")
    food = state.get("food_preference")
    travelers = state.get("num_travelers")
    city = state.get("current_city")
    budget = state.get("total_budget")

    if airline:
        parts.append(f"- Preferred airline: **{airline}**")
    if food:
        parts.append(f"- Food preference: **{food}**")
    if travelers:
        parts.append(f"- Number of travelers: **{travelers}**")
    if city:
        parts.append(f"- Last destination: **{city}**")
    if budget:
        parts.append(f"- Last budget: **${budget:,.0f}**")

    if parts:
        content = "Here's what I remember about you:\n" + "\n".join(parts)
    else:
        content = (
            "I don't have any saved preferences yet. "
            "Tell me your preferred airline, dietary needs, or how many people are traveling!"
        )

    logger.info("Recall node answered from saved state.")
    return {"messages": [AIMessage(content=content)]}


# ── Node 4: Validator ─────────────────────────────────────────────────────────

def run_validator(state: AgentState) -> dict:
    """
    Security guardrail node — runs before Marco on every user message.

    Checks for:
      - Prompt injection attempts
      - Out-of-scope requests (not travel related)
      - Unsupported destinations (not in the database)

    Writes validation_status into State so the router can decide
    whether to pass the message to Marco or terminate immediately.
    If blocked, injects a rejection AIMessage so the user sees a clear reason.
    """
    from src.agents.validator import validate_input

    messages = state.get("messages", [])
    if not messages:
        return {"validation_status": "approved"}

    last_content = getattr(messages[-1], "content", "")
    result = validate_input(last_content)

    logger.info(
        "Validator: verdict=%s reason=%s", result.verdict, result.reason
    )

    if not result.approved:
        rejection = AIMessage(content=result.rejection_message)
        return {
            "validation_status": result.verdict.lower(),
            "messages": [rejection],
        }

    return {"validation_status": "approved"}


# ── Node 3: Agent (LLM call) ──────────────────────────────────────────────────

def call_model(state: AgentState) -> dict:
    """
    Core agent node — sends the conversation history to the LLM and returns
    either a tool-call request or a final human-readable answer.
    Increments tool_call_count whenever a tool is requested.

    Automatically retries on 429 rate-limit errors using the delay
    reported by the API itself (free tier: 5 req/min on gemini-2.5-flash).
    """
    # ── Build dynamic additions to the system prompt ─────────────────────────
    profile_lines = []
    if state.get("preferred_airline"):
        profile_lines.append(f"- Preferred airline: {state['preferred_airline']}")
    if state.get("food_preference"):
        profile_lines.append(f"- Dietary preference: {state['food_preference']}")
    if state.get("num_travelers"):
        profile_lines.append(f"- Traveling with: {state['num_travelers']} people")

    summary = state.get("conversation_summary", "")

    # Prompt caching: reuse the cached _BASE_SYSTEM_MSG when nothing extra is needed.
    # Only construct a new SystemMessage when profile or summary data exists.
    if profile_lines or summary:
        extra = ""
        if profile_lines:
            extra += (
                "\n\n## User Profile (remembered from previous sessions)\n"
                + "\n".join(profile_lines)
                + "\nAlways apply these preferences when recommending flights, hotels, and activities."
            )
        if summary:
            extra += f"\n\n## Conversation Summary (past context)\n{summary}"
        system_msg = SystemMessage(content=PLANNER_SYSTEM_PROMPT + extra)
    else:
        system_msg = _BASE_SYSTEM_MSG  # cached — no reconstruction needed

    # ── Compact memory: limit messages sent to LLM when history is long ───────
    all_messages = state["messages"]
    if summary and len(all_messages) > 10:
        # Send only the 8 most recent messages; older context is in the summary
        messages_to_send = all_messages[-8:]
        logger.info(f"Compact memory: sending {len(messages_to_send)}/{len(all_messages)} messages to LLM.")
    else:
        messages_to_send = all_messages

    messages = [system_msg] + messages_to_send

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
                time.sleep(wait)
            else:
                raise

    count = state.get("tool_call_count", 0)
    if hasattr(response, "tool_calls") and response.tool_calls:
        count += len(response.tool_calls)
        names = [tc["name"] for tc in response.tool_calls]
        logger.info(f"Tool calls +{len(response.tool_calls)} (total {count}): {names}")

    return {"messages": [response], "tool_call_count": count}


# ── Node 4: Circuit Breaker ───────────────────────────────────────────────────

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


# ── Node 5: Researcher ────────────────────────────────────────────────────────

def researcher_node(state: AgentState) -> dict:
    """
    Lightweight data-lookup node — handles simple factual queries
    (e.g. "what hotels are in Tokyo?") by calling DB tools directly.
    Routes here when route_after_validator detects a research-only question.
    Bypasses the full planning workflow to avoid unnecessary LLM calls.
    """
    import json
    from src.tools.db_tools import fetch_hotels, fetch_flights, fetch_activities

    last_content = getattr(state["messages"][-1], "content", "").lower()
    city = state.get("current_city", "")

    if "hotel" in last_content and city:
        raw = fetch_hotels.invoke({"city": city})
        label = f"Hotels in {city}"
        content = _format_hotels(label, raw)
    elif "flight" in last_content and city:
        raw = fetch_flights.invoke({"origin": "TLV", "destination": city})
        label = f"Flights from TLV to {city}"
        content = _format_flights(label, raw)
    elif "activit" in last_content and city:
        raw = fetch_activities.invoke({"city": city})
        label = f"Activities in {city}"
        content = _format_activities(label, raw)
    else:
        content = "Please specify hotels, flights, or activities along with a city name."

    logger.info("Researcher node completed query for city=%s.", city or "unknown")
    return {"messages": [AIMessage(content=content)]}


def _format_hotels(label: str, raw: str) -> str:
    import json
    try:
        items = json.loads(raw)
    except (ValueError, TypeError):
        return f"**{label}:**\n{raw}"
    lines = [f"**{label}:**\n"]
    for h in items:
        stars = "★" * h.get("stars", 0)
        lines.append(f"- **{h['name']}** {stars} — ${h['price_per_night']}/night")
    return "\n".join(lines)


def _format_flights(label: str, raw: str) -> str:
    import json
    try:
        items = json.loads(raw)
    except (ValueError, TypeError):
        return f"**{label}:**\n{raw}"
    if isinstance(items, dict):
        items = [items]
    lines = [f"**{label}:**\n"]
    for f in items:
        lines.append(f"- **{f['airline']}** ({f['flight_number']}) — ${f['price']}")
    return "\n".join(lines)


def _format_activities(label: str, raw: str) -> str:
    import json
    try:
        items = json.loads(raw)
    except (ValueError, TypeError):
        return f"**{label}:**\n{raw}"
    lines = [f"**{label}:**\n"]
    for a in items:
        price = f"${a['price']}" if a.get("price") else "Free"
        lines.append(f"- **{a['name']}** ({a.get('category', '')}) — {price}")
    return "\n".join(lines)


# ── Node 6: Reviewer ──────────────────────────────────────────────────────────

def reviewer_node(state: AgentState) -> dict:
    """
    Quality-control node — automatically critiques Marco's final travel plan.
    Only runs when the agent produced a full plan (city detected + 5+ tool calls).
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


# ── Node 10: Summarizer ───────────────────────────────────────────────────────

def summarizer_node(state: AgentState) -> dict:
    """
    Compact memory node — runs after every completed conversation turn.

    When message history exceeds 10 messages it asks the LLM to produce a
    concise bullet-point summary of the older messages, stores it in
    conversation_summary, and limits future LLM calls to the last 8 messages.
    For short sessions (≤10 messages) it is a no-op and returns immediately.
    """
    messages = state.get("messages", [])

    if len(messages) <= 10:
        return {}

    # Summarize all but the 4 most recent messages
    to_summarize = messages[:-4]

    transcript_lines = []
    for msg in to_summarize:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        transcript_lines.append(f"{role}: {content[:300]}")

    if not transcript_lines:
        return {}

    summary_response = _get_model().invoke([
        SystemMessage(content="You are a concise conversation summarizer."),
        HumanMessage(
            content=(
                "Summarize this travel planning conversation in 3–4 bullet points. "
                "Include: destinations discussed, budgets, user preferences, and key decisions.\n\n"
                + "\n".join(transcript_lines)
            )
        ),
    ])

    summary = (
        summary_response.content
        if isinstance(summary_response.content, str)
        else str(summary_response.content)
    )

    logger.info(f"Summarizer: compressed {len(to_summarize)} messages → summary ({len(summary)} chars).")
    return {"conversation_summary": summary}


# ── Tool Node (prebuilt) ──────────────────────────────────────────────────────

def build_tools_node():
    from src.tools import ALL_TOOLS
    return ToolNode(ALL_TOOLS)
