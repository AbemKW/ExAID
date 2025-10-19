from langchain_openai import OpenAI
from langchain.tools import tool

@tool
def summarize_text(text: str) -> str:
    # Implement your text summarization logic here
    return "Summary of the text"
