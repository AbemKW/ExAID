from langchain_openai import OpenAI
from langchain.tools import tool
from tools.llm import llm


def summarize(input: str) -> str:
    """Summarizes the given text input."""
    prompt = f"Summarize this text into 1 sentence:\n\n{input}"
    response = llm.invoke(prompt)
    return response