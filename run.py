"""
Convenience launcher — run from the project root: python run.py
"""

# Load .env BEFORE any src imports so the API key is in the environment
# when ChatGoogleGenerativeAI is first constructed inside the graph.
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from src.main import run

if __name__ == "__main__":
    run()
