"""
Interactive CLI entry point for the AI Travel Planner.

Run:  python run.py
      python -m src.main

Commands during a session:
  'exit' — end the session
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
║  Full plans are auto-reviewed  |  'exit' to quit         ║
╚══════════════════════════════════════════════════════════╝
"""

_SUPPORTED = "Supported cities: Paris · London · Tokyo · New York · Berlin"


def _extract_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def run() -> None:
    config = {"configurable": {"thread_id": "session_01"}}

    print(_BANNER)
    print(_SUPPORTED + "\n")

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

        # ── Stream through the graph, printing each AI response as it arrives ─
        seen_contents: set = set()

        for event in graph.stream(
            {"messages": [("user", user_input)]},
            config,
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]

            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                names = ", ".join(tc["name"] for tc in last_msg.tool_calls)
                print(f"  [calling: {names}]", flush=True)

            elif (
                isinstance(last_msg, AIMessage)
                and last_msg.content
                and not getattr(last_msg, "tool_calls", None)
            ):
                text = _extract_text(last_msg.content)
                if text and text not in seen_contents:
                    seen_contents.add(text)
                    print(f"\nAgent: {text}\n")

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