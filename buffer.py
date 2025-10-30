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
        self.buffer= {}
        self.flag_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are monitoring the reasoning streams of multiple AI agents. "
            "For the specified agent, compare the new reasoning snippet to the previous context for that agent. "
            "If the topic or reasoning goal has changed, or if a complete thought appears to end, reply with exactly 'YES'. Otherwise, reply 'NO'. "
            "Always consider the agent's identity and context in your analysis."),
            ("user", "Accumulated reasoning traces in buffer for {agent_id}:\n{previous_trace}\n\nNew trace:\n{new_trace}"),
            ("user", "Accumulated reasoning traces in buffer for all agents: \n{all_previous_traces}")
        ])

    async def addchunk(self, agent_id: str, chunk: str) :
        if agent_id not in self.buffer:
            self.buffer[agent_id] = []

        self.buffer[agent_id].append(chunk)

        flag_chain = self.flag_prompt | llm
        flag_response = await flag_chain.ainvoke({
            "previous_trace": "\n".join(self.buffer[agent_id]),
            "new_trace": chunk,
            "all_previous_traces": "\n".join(
                [f"| {aid} |\n" + "\n".join(chunks) for aid, chunks in self.buffer.items()]
            )
        })
        return "YES" in flag_response.upper()
    
    def flush(self) -> list[str]:
        flushed = self.buffer.copy()
        self.buffer.clear()
        return flushed
            