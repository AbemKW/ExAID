from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen/qwen3-4b-2507",
    base_url="https://92cb1d692ca2.ngrok-free.app/v1",
    api_key="dummykey"
)
summarizer = ChatOpenAI(
    model="qwen/qwen3-4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="dummykey"
)