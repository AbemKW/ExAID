from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen/qwen3-4b-2507",
    base_url="https://d83d72d3cba9.ngrok-free.app/v1",
    api_key="dummykey"
)