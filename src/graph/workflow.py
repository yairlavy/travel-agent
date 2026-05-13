import sqlite3
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.graph.state import AgentState
from src.graph.nodes import (
    extract_metadata, update_preferences, recall_node,
    run_validator, call_model, circuit_breaker,
    researcher_node, reviewer_node, summarizer_node, build_tools_node,
)
from src.graph.router import route_after_metadata, route_after_validator, should_continue

"""
Graph topology
──────────────
START
  │
  ▼
extract_metadata          ← resets counter, detects city/budget
  │
  ▼  route_after_metadata (conditional edge)
  ├─ [recall]             → recall_node ─────────────────────────────→ END
  ├─ [update_preferences] → update_preferences ──────────────────────→ validator
  └─ [normal]             → validator
                               │
                               ├─ [blocked]        → END
                               ├─ [research query] → researcher ──────→ END
                               └─ [approved]       → agent ◄──────────┐
                                                        │              │
                                                        ├─ tools ──────┘  (loop)
                                                        ├─ circuit_breaker → END
                                                        │
                                  ┌─────────────────────┤
                                  │  is_admin + 5+ tools │
                                  ▼                     │
                              reviewer                  │
                                  │                     │
                                  └──────┐   ┌──────────┘
                                         ▼   ▼
                                       summarizer → END
"""

# SQLite connection created once at module level — stays open for app lifetime
_DB_PATH = Path(__file__).parent.parent.parent / "data" / "checkpoints.db"
_conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
_checkpointer = SqliteSaver(_conn)


def build_graph():
    """
    Compile and return the StateGraph with SqliteSaver for persistent
    cross-session memory. Each unique thread_id is a separate conversation.

    Admin sessions (thread_id ending with ADMIN00) route through the reviewer
    node after full plans. All sessions route through the summarizer for
    compact memory management.
    """
    builder = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node("extract_metadata", extract_metadata)
    builder.add_node("update_preferences", update_preferences)
    builder.add_node("recall", recall_node)
    builder.add_node("validator", run_validator)
    builder.add_node("researcher", researcher_node)
    builder.add_node("agent", call_model)
    builder.add_node("tools", build_tools_node())
    builder.add_node("circuit_breaker", circuit_breaker)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("summarizer", summarizer_node)          # NEW

    # ── Edges ──────────────────────────────────────────────────────────────────
    builder.add_edge(START, "extract_metadata")

    builder.add_conditional_edges(
        "extract_metadata", route_after_metadata,
        {"recall": "recall", "update_preferences": "update_preferences", "validator": "validator"},
    )

    builder.add_edge("recall", END)
    builder.add_edge("update_preferences", "summarizer")  # ack + done, no agent needed

    builder.add_conditional_edges(
        "validator", route_after_validator,
        {"researcher": "researcher", "agent": "agent", END: END},
    )
    builder.add_edge("researcher", END)

    # should_continue now routes to "summarizer" instead of END for final answers
    builder.add_conditional_edges(
        "agent", should_continue,
        {"tools": "tools", "circuit_breaker": "circuit_breaker",
         "reviewer": "reviewer", "summarizer": "summarizer"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("circuit_breaker", END)

    # reviewer → summarizer (admin full plans get reviewed then summarized)
    builder.add_edge("reviewer", "summarizer")               # changed from reviewer → END

    # summarizer is the final step for all normal answers
    builder.add_edge("summarizer", END)

    return builder.compile(checkpointer=_checkpointer)


# Module-level singleton used by main.py
graph = build_graph()
