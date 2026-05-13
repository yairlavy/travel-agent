from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import AgentState
from src.graph.nodes import extract_metadata, call_model, circuit_breaker, build_tools_node, run_validator
from src.graph.router import should_continue, route_after_validator

"""
Graph topology
──────────────
START
  │
  ▼
extract_metadata          ← resets counter, detects city/budget (Node 1)
  │
  ▼
validator                 ← security guardrail (Node 2)
  │
  ├─ [blocked]            → END  (injection / out of scope / unsupported city)
  │
  └─ [approved]
        │
        ▼
      agent  ◄────────────────────────────────────────────┐
        │                                                  │
        ├─ [tool_calls present & count < MAX] → tools ─────┘  (loop)
        │
        ├─ [count >= MAX or repetition]       → circuit_breaker → END
        │
        └─ [no tool_calls]                    → END
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

    # ── Nodes ────────────────────────────────────────────────────────────────
    builder.add_node("extract_metadata", extract_metadata)
    builder.add_node("validator", run_validator)
    builder.add_node("agent", call_model)
    builder.add_node("tools", build_tools_node())
    builder.add_node("circuit_breaker", circuit_breaker)

    # ── Edges ─────────────────────────────────────────────────────────────────
    builder.add_edge(START, "extract_metadata")                       # always start here
    builder.add_edge("extract_metadata", "validator")                 # fixed: metadata → validator
    builder.add_conditional_edges("validator", route_after_validator) # approved → agent | blocked → END
    builder.add_conditional_edges("agent", should_continue)           # branch based on output
    builder.add_edge("tools", "agent")                                # loop back after tool execution
    builder.add_edge("circuit_breaker", END)                          # safety exit

    checkpointer = MemorySaver() if use_memory else None
    return builder.compile(checkpointer=checkpointer)


# Module-level singleton used by main.py and agents
graph = build_graph()
