from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import AgentState
from src.graph.nodes import (
    extract_metadata, run_validator, call_model, circuit_breaker,
    researcher_node, reviewer_node, build_tools_node,
)
from src.graph.router import route_after_validator, should_continue

"""
Graph topology
──────────────
START
  │
  ▼
extract_metadata          ← resets counter, detects city/budget
  │
  ▼
validator                 ← security guardrail (injection / scope / city check)
  │
  ├─ [blocked]            → END  (rejection message already injected)
  │
  ├─ [research query]     → researcher ──────────────────────────────→ END
  │
  └─ [approved + plan]    → agent  ◄────────────────────────────────┐
                               │                                     │
                               ├─ [tool_calls & count < MAX] → tools┘  (loop)
                               │
                               ├─ [count >= MAX or repetition] → circuit_breaker → END
                               │
                               ├─ [final answer + city + 5+ tools] → reviewer → END
                               │
                               └─ [simple answer] → END
"""


def build_graph(use_memory: bool = True):
    """
    Compile and return the StateGraph.

    use_memory=True  → MemorySaver (in-session persistence, default)
    use_memory=False → stateless (useful for testing single turns)

    For persistent cross-session memory (Session 4), swap MemorySaver for:
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver.from_conn_string("data/checkpoints.db")
    """
    builder = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node("extract_metadata", extract_metadata)
    builder.add_node("validator", run_validator)
    builder.add_node("researcher", researcher_node)
    builder.add_node("agent", call_model)
    builder.add_node("tools", build_tools_node())
    builder.add_node("circuit_breaker", circuit_breaker)
    builder.add_node("reviewer", reviewer_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    builder.add_edge(START, "extract_metadata")
    builder.add_edge("extract_metadata", "validator")
    builder.add_conditional_edges(
        "validator", route_after_validator,
        {"researcher": "researcher", "agent": "agent", END: END},
    )
    builder.add_edge("researcher", END)
    builder.add_conditional_edges("agent", should_continue)
    builder.add_edge("tools", "agent")
    builder.add_edge("circuit_breaker", END)
    builder.add_edge("reviewer", END)

    checkpointer = MemorySaver() if use_memory else None
    return builder.compile(checkpointer=checkpointer)


# Module-level singleton used by main.py and agents
graph = build_graph()
