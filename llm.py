from langchain_openai import OpenAI

llm = OpenAI(
    model="qwen/qwen3-4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="dummykey"
)