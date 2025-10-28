from langchain_openai import OpenAI

llm = OpenAI(
    model="qwen/qwen3-4b-2507",
    base_url="https://ed7a5a297b8b.ngrok-free.app/v1",
    api_key="dummykey"
)