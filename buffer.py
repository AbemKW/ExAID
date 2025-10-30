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
            "For the specified agent, compare the new reasoning snippet to the previous context for that agent. "
            "If the topic or reasoning goal has changed, or if a complete thought appears to end, reply with exactly 'YES'. Otherwise, reply 'NO'. "
            "Always consider the agent's identity and context in your analysis."),
            ("user", "Accumulated reasoning traces in buffer:\n{previous_trace}\n\nNew trace:\n{new_trace}"),
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
            