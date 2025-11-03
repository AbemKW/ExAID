from langchain_core.prompts import ChatPromptTemplate
from llm import llm
import queue

class TraceBuffer:
    """A tiny buffer that stores chunks per-agent.

    When a given agent's accumulated chunks reach `chunk_threshold`, the
    `on_full_callback` is invoked with two arguments: (agent_id, combined_text),
    and that agent's buffer is reset.
    """

    def __init__(self):
        self.buffer: list[str] = []
        self.flag_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are monitoring the reasoning streams of multiple AI agents. "
            "Your task is to decide if the new reasoning trace should trigger summarization. "
            "Reply with EXACTLY 'YES' (all caps) if ANY of these conditions are met:\n"
            "- The new trace completes a thought or reasoning step\n"
            "- The topic or focus has changed from previous traces\n"
            "- Enough context has accumulated to warrant a summary\n"
            "- The reasoning has reached a natural pause or conclusion\n\n"
            "Reply with EXACTLY 'NO' (all caps) if the new trace is just continuing the same line of reasoning without completion.\n"
            "Be decisive - prefer YES when in doubt, as summaries help track agent progress."),
            ("user", "Previous traces in buffer:\n{previous_trace}\n\nNew trace to evaluate:\n{new_trace}\n\nShould this trigger summarization? Reply with only 'YES' or 'NO'."),
        ])

    async def addchunk(self, agent_id: str, chunk: str) :

        tagged_chunk = f"| {agent_id} | {chunk}"
        self.buffer.append(tagged_chunk)

        flag_chain = self.flag_prompt | llm
        flag_response = await flag_chain.ainvoke({
            "previous_trace": "\n".join(self.buffer),
            "new_trace": tagged_chunk
        })
        return "YES" in flag_response.upper()
    
    def flush(self) -> list[str]:
        flushed = self.buffer.copy()
        self.buffer.clear()
        return flushed
            