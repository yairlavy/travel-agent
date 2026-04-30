"""
Tests Gemini API connectivity.
Skipped automatically when GOOGLE_API_KEY is not set (e.g. in CI without the secret).
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()


def test_google_api_key_present():
    assert os.getenv("GOOGLE_API_KEY"), (
        "GOOGLE_API_KEY is not set. Add it to your .env file."
    )


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not available",
)
def test_gemini_responds():
    from src.agents.base import get_model

    model = get_model(temperature=0)
    response = model.invoke("Reply with exactly one word: OK")
    assert response.content.strip() != "", "Model returned an empty response"
    print(f"\nGemini response: {response.content.strip()}")
