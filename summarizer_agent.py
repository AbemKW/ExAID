from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from llm import llm
from summary_state import SummaryState as State


summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "{input} \n\nSummarize this text using layered approach. "
    "First, identify three key points which are facts, implications, and results."
    "Then condense them into a concise summary. "
    "Ensure clarity and coherence in the final output."),
    ("user", "Address the agent as {agent_id}")
]
)

summarize_chain = summarize_prompt | llm

def summarize(agent_id: str, text: str, graph: State) -> str:
    """Summarizes the given text input."""
    graph.summarizer_busy = True
    summary = summarize_chain.invoke({
        "input": text, 
        "agent_id": agent_id})
    graph.summarizer_busy = False
    return summary, graph