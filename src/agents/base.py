import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL_NAME = os.getenv("LLM_MODEL", "gemini-2.5-flash")


def get_model(
    temperature: float = 0,
    bind_tools: list = None,
) -> ChatGoogleGenerativeAI:
    """
    Factory — returns a configured Gemini model instance.
    Pass bind_tools to create a tool-capable model for agent nodes.
    All agents share this factory so the model name stays in one place.
    """
    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
    )
    if bind_tools:
        return model.bind_tools(bind_tools)
    return model
