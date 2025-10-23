from summary_state import SummaryState
from buffer import TraceBuffer
from summarizer_agent import summarize

class EXAID:
    def __init__(self,chunk_threshold: int = 5):
        self.graph = SummaryState()
        self.agents = {}
        self.buffer = TraceBuffer(self._on_buffer_full,chunk_threshold)

    def addAgent(self, agent, id: str):
        self.agents[id] = agent
        self.graph.add_agent(id)

    async def addTrace(self, id: str, text: str):
        # pass agent id and text separately to the buffer
        self.buffer.addchunk(id, text)

    async def _on_buffer_full(self, agent_id: str, combined_text: str):
        # store trace and produce summary for the given agent
        self.graph.add_trace(agent_id, combined_text)
        # summarizer is synchronous in this project; call directly
        summary = summarize(combined_text)
        self.graph.add_summary(agent_id, summary)
