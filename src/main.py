"""
Interactive CLI entry point for the AI Travel Planner.

Run:  python run.py
      python -m src.main

Commands during a session:
  'review'  — ask the reviewer agent to critique the last plan
  'exit'    — end the session
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve .env relative to the project root (two levels up from src/main.py)
load_dotenv(Path(__file__).parent.parent / ".env")

_api_key = os.getenv("GOOGLE_API_KEY", "")
if not _api_key or _api_key.startswith("your_"):
    raise SystemExit(
        "\n[ERROR] GOOGLE_API_KEY is not set.\n"
        "Edit the .env file in the project root and add your real key:\n\n"
        "  GOOGLE_API_KEY=AIza...\n\n"
        "Get a free key at: https://aistudio.google.com/apikey\n"
    )

from langchain_core.messages import AIMessage
from src.graph.workflow import graph
from src.utils.logger import get_logger

logger = get_logger("main")

_BANNER = """
╔══════════════════════════════════════════════════════════╗
║        AI Travel Planner  •  LangGraph + Gemini          ║
║  Commands: 'review' = critique last plan | 'exit' = quit ║
╚══════════════════════════════════════════════════════════╝
"""

_SUPPORTED = "Supported cities: Paris · London · Tokyo · New York · Berlin"


def run() -> None:
    config = {"configurable": {"thread_id": "session_01"}}

    print(_BANNER)
    print(_SUPPORTED + "\n")

    last_plan: str | None = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! Safe travels!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! Safe travels!")
            break

        # ── Special command: review the last plan ─────────────────────────
        if user_input.lower() == "review":
            if not last_plan:
                print("Agent: No plan to review yet. Ask for a trip first.\n")
                continue
            from src.agents.reviewer import review_plan
            print("\nReviewer:")
            print(review_plan(last_plan))
            print()
            continue

        # ── Regular query: stream through the graph ───────────────────────
        final_response: str | None = None

        for event in graph.stream(
            {"messages": [("user", user_input)]},
            config,
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]

            # Show live tool-call hints so the user knows work is happening
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                names = ", ".join(tc["name"] for tc in last_msg.tool_calls)
                print(f"  [calling: {names}]", flush=True)

            # Capture the final AI text response (not a ToolMessage)
            elif (
                isinstance(last_msg, AIMessage)
                and last_msg.content
                and not getattr(last_msg, "tool_calls", None)
            ):
                final_response = last_msg.content

        if final_response:
            print(f"\nAgent: {final_response}\n")
            last_plan = final_response
            logger.info(
                "turn complete | city=%s | budget=%s | tools=%d",
                event.get("current_city", "—"),
                event.get("total_budget", "—"),
                event.get("tool_call_count", 0),
            )


if __name__ == "__main__":
    db_path = Path(__file__).parent.parent / "data" / "travel_agency.db"
    if not db_path.exists():
        print("First run — initializing database...")
        from src.utils.db_init import create_travel_db
        create_travel_db()
    run()
