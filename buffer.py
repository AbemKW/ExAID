from langchain_core.prompts import ChatPromptTemplate
from summary_state import SummaryState
from llm import llm
import queue

class TraceBuffer:
    """A tiny buffer that stores chunks per-agent.

    When a given agent's accumulated chunks reach `chunk_threshold`, the
    `on_full_callback` is invoked with two arguments: (agent_id, combined_text),
    and that agent's buffer is reset.
    """

    def __init__(self, on_full_callback, graph: SummaryState):
        # map agent_id -> list[str]
        self.buffer: dict[str, list[str]] = {}
        self.on_full_callback = on_full_callback
        self.graph = graph
        self.queue = queue.Queue(-1)  # Global queue for all agents: (agent_id, combined_text)

    async def addchunk(self, agent_id: str, chunk: str):
        # initialize list for agent if necessary
        if agent_id not in self.buffer:
            self.buffer[agent_id] = []

        flag_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are monitoring an AI agent's reasoning stream. "
            "Compare the new reasoning snippet to the previous context. "
            "If the topic or reasoning goal has changed, or if a complete thought "
            "appears to end, reply with exactly 'YES'. Otherwise, reply 'NO'."),
            ("user", "Previous context:\n{previous_trace}\n\nNew snippet:\n{new_trace}"),
        ])
        flag_chain = flag_prompt | llm
        flag_response = await flag_chain.ainvoke({"previous_trace": self.buffer[agent_id], "new_trace": chunk})
        if "YES" in flag_response.upper():
            # If topic has shifted or thought completed, flush current buffer first
            if self.buffer[agent_id]:
                combined = "\n".join(self.buffer[agent_id])
                try:
                    if not self.graph.summarizer_busy:
                        await self.on_full_callback(agent_id, combined)
                    else:
                        self.queue.put((agent_id, combined))
                except Exception as e:
                    # Log the error and continue; buffer will still be reset
                    print(f"Error in on_full_callback for agent {agent_id}: {e}")
                finally:
                    self.buffer[agent_id] = []
        elif "NO" in flag_response.upper():
            # If topic has not shifted, just append the chunk
            self.buffer[agent_id].append(chunk)