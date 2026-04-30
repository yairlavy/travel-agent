"""
Researcher agent — a lightweight, data-only agent for focused lookups.

Useful when you want raw facts (e.g. "what hotels exist in Tokyo?")
without the full planning workflow. Called directly as a function,
not as a LangGraph node.
"""

from langchain_core.messages import SystemMessage
from src.agents.base import get_model
from src.tools.db_tools import (
    fetch_flights,
    fetch_hotels,
    fetch_activities,
    get_visa_requirement,
)

_RESEARCHER_PROMPT = """You are a travel data researcher. Your sole job is to retrieve
and present factual, structured information about destinations.

Rules:
- Only answer from data you retrieve with the provided tools.
- Present numbers and specifics — never guess or estimate.
- If data is missing, say so clearly and stop.
- Keep responses concise and well-formatted.
"""

_tools = [fetch_flights, fetch_hotels, fetch_activities, get_visa_requirement]
_model = get_model(temperature=0, bind_tools=_tools)


def research(query: str) -> str:
    """
    Run a focused, single-turn research query against the travel database.
    Returns the model's response as a plain string.
    """
    response = _model.invoke([
        SystemMessage(content=_RESEARCHER_PROMPT),
        ("user", query),
    ])
    return response.content
