from langchain_core.prompts import ChatPromptTemplate
from llm import llm



summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarizer for a multi-agent reasoning system. "
    "Given the summary history, the latest summary, and a new reasoning buffer, respond only with the new information as a single, concise sentence. "
    "Do not repeat or rewrite the previous summary. Do not include any instructions, explanations, or formatting. Your response must be only the new information as a single, clear sentence."),
    ("user", "Summary history:\n[ {summary_history} ]\n\nLatest summary:\n{latest_summary}\n\nNew reasoning buffer:\n{new_buffer}\n\nUpdate the summary accordingly."),
]
)
summarize_chain = summarize_prompt | llm

async def summarize(summary_history: list[str], latest_summary: str, new_buffer: str) -> str:
    """Updates the summary given the summary history (as a list), latest summary, and new reasoning buffer."""
    summary = await summarize_chain.ainvoke({
        "summary_history": ",\n".join(summary_history),
        "latest_summary": latest_summary,
        "new_buffer": new_buffer
    })
    return summary