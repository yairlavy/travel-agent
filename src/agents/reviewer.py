"""
Reviewer agent — self-correction and quality control.

After the planner produces a travel plan, the reviewer critiques it
for completeness, budget realism, and missing information.
Called from main.py via the 'review' command, not as a graph node.
"""

from langchain_core.messages import SystemMessage
from src.agents.base import get_model

_REVIEWER_PROMPT = """You are a critical travel plan reviewer and quality controller.

Evaluate the provided travel plan on these dimensions:
1. Budget realism     — are cost estimates accurate and achievable?
2. Completeness       — are flights, accommodation, and activities covered?
3. Visa & legal       — are entry requirements explicitly mentioned?
4. Practical gaps     — what is missing, unclear, or could go wrong?

Output format:
- Score: X/10
- Strengths: (2–3 bullet points)
- Issues: (bullet points of gaps or errors)
- Suggestions: (2–3 concrete improvements)

Be constructive, specific, and honest. Do not pad your response.
"""

_model = get_model(temperature=0.3)


def review_plan(plan: str) -> str:
    """
    Critique and score a travel plan string.
    Returns the reviewer's structured feedback as a plain string.
    """
    response = _model.invoke([
        SystemMessage(content=_REVIEWER_PROMPT),
        ("user", f"Please review this travel plan:\n\n{plan}"),
    ])
    return response.content
