"""
Rich terminal UI entry point for the AI Travel Planner.

Run:  python run_rich.py

Renders agent responses in styled panels, shows live tool-call
progress, and displays the auto-review in a separate coloured panel.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_api_key = os.getenv("GOOGLE_API_KEY", "")
if not _api_key or _api_key.startswith("your_"):
    raise SystemExit(
        "\n[ERROR] GOOGLE_API_KEY is not set.\n"
        "Edit the .env file in the project root and add your real key.\n"
    )

from langchain_core.messages import AIMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme
from rich.prompt import Prompt
from rich.columns import Columns
from typing import Optional

from src.graph.workflow import graph
from src.utils.logger import get_logger

logger = get_logger("main_rich")

_THEME = Theme({
    "user.label":     "bold cyan",
    "user.text":      "cyan",
    "agent.border":   "green",
    "reviewer.border":"yellow",
    "researcher.border": "blue",
    "tool.call":      "dim white",
    "status.city":    "bold magenta",
    "status.budget":  "bold green",
    "banner":         "bold white on dark_green",
    "exit.hint":      "dim",
})

console = Console(theme=_THEME, highlight=False)

_SUPPORTED = "Paris  ·  London  ·  Tokyo  ·  New York  ·  Berlin"


def _print_banner() -> None:
    console.print()
    console.print(Panel(
        Text.assemble(
            ("  AI Travel Planner\n", "bold white"),
            ("  Powered by Gemini + LangGraph\n\n", "dim white"),
            ("  Destinations: ", "dim white"),
            (_SUPPORTED, "bold cyan"),
        ),
        title="[bold white]✈  Marco[/bold white]",
        border_style="green",
        padding=(0, 2),
    ))
    console.print(Text("  Type 'exit' to quit.", style="exit.hint"))
    console.print()


def _print_user(text: str) -> None:
    console.print(Panel(
        Text(text, style="user.text"),
        title="[user.label]You[/user.label]",
        border_style="cyan",
        padding=(0, 2),
    ))


def _extract_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _is_review(text: str) -> bool:
    return text.startswith("\n---\n**Plan Review")


def _is_researcher(text: str) -> bool:
    return text.startswith("**Hotels in") or text.startswith("**Flights") or text.startswith("**Activities")


def _print_agent(text: str) -> None:
    if _is_review(text):
        clean = text.replace("\n---\n**Plan Review (auto):**\n", "").strip()
        console.print(Panel(
            Markdown(clean),
            title="[yellow]✦ Auto Review[/yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))
    elif _is_researcher(text):
        console.print(Panel(
            Markdown(text),
            title="[blue]⚡ Quick Lookup[/blue]",
            border_style="blue",
            padding=(1, 2),
        ))
    else:
        console.print(Panel(
            Markdown(text),
            title="[green]✈  Marco[/green]",
            border_style="green",
            padding=(1, 2),
        ))


def _print_status(city: Optional[str], budget: Optional[float], tool_count: int) -> None:
    if not city and not budget:
        return
    parts = []
    if city:
        parts.append(Text.assemble(("Destination: ", "dim"), (city, "status.city")))
    if budget:
        parts.append(Text.assemble(("Budget: $", "dim"), (f"{budget:,.0f}", "status.budget")))
    if tool_count:
        parts.append(Text(f"Tools used: {tool_count}", style="dim"))
    console.print(Rule(style="dim"))
    console.print(Columns(parts, padding=(0, 4)))
    console.print()


def run() -> None:
    config = {"configurable": {"thread_id": "session_01"}}

    _print_banner()

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! Safe travels! ✈[/dim]\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            console.print("\n[dim]Goodbye! Safe travels! ✈[/dim]\n")
            break

        console.print()

        seen_contents: set = set()
        last_event: dict = {}

        _TOOL_LABELS = {
            "fetch_flights":       "✈  Searching flights",
            "fetch_hotels":        "🏨  Searching hotels",
            "fetch_activities":    "🎭  Finding activities",
            "get_visa_requirement":"🛂  Checking visa requirements",
            "calculate_trip_cost": "💰  Calculating costs",
            "get_cheapest_flight": "✈  Finding cheapest flight",
            "get_cheapest_hotel":  "🏨  Finding cheapest hotel",
            "list_destinations":   "🗺  Listing destinations",
            "web_search":          "🌐  Searching the web",
        }

        try:
            with console.status("[tool.call]Marco is thinking...[/tool.call]", spinner="dots") as status:
                for event in graph.stream(
                    {"messages": [("user", user_input)]},
                    config,
                    stream_mode="values",
                ):
                    last_event = event
                    last_msg = event["messages"][-1]

                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        labels = [
                            _TOOL_LABELS.get(tc["name"], f"⚙  {tc['name']}")
                            for tc in last_msg.tool_calls
                        ]
                        status.update(f"[tool.call]{' · '.join(labels)}...[/tool.call]")

                    elif (
                        isinstance(last_msg, AIMessage)
                        and last_msg.content
                        and not getattr(last_msg, "tool_calls", None)
                    ):
                        text = _extract_text(last_msg.content)
                        if text and text not in seen_contents:
                            seen_contents.add(text)
                            status.stop()
                            _print_agent(text)
                            status.start()
                            status.update("[tool.call]✦  Reviewing plan...[/tool.call]")

            _print_status(
                city=last_event.get("current_city"),
                budget=last_event.get("total_budget"),
                tool_count=last_event.get("tool_call_count", 0),
            )

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                import re
                wait = re.search(r"retry in (\d+)", err)
                wait_msg = f"Retry in {wait.group(1)}s." if wait else "Try again in a minute."
                console.print(Panel(
                    f"[yellow]You've hit the Gemini free-tier quota limit.\n{wait_msg}[/yellow]\n\n"
                    "[dim]The free tier allows 20 requests/day and 10/minute.\n"
                    "Wait a moment and try again, or use a simpler query.[/dim]",
                    title="[red]Rate Limited[/red]",
                    border_style="red",
                ))
            else:
                console.print(Panel(
                    f"[red]Unexpected error:[/red] {err[:300]}",
                    title="[red]Error[/red]",
                    border_style="red",
                ))

        logger.info(
            "turn complete | city=%s | budget=%s | tools=%d",
            last_event.get("current_city", "—"),
            last_event.get("total_budget", "—"),
            last_event.get("tool_call_count", 0),
        )


if __name__ == "__main__":
    db_path = Path(__file__).parent.parent / "data" / "travel_agency.db"

    if not db_path.exists():
        console.print("[dim]First run — initializing database...[/dim]")
        from src.utils.db_init import create_travel_db
        create_travel_db()

    run()
