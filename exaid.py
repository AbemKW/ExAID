from summary_state import SummaryState
from buffer import TraceBuffer
from summarizer_agent import summarize
class EXAID:
    def __init__(self,chunk_threshold: int = 5):
        self.name = "EXAID"
        self.graph = SummaryState()
        self.agents = {}
        self.buffer = TraceBuffer(self._on_buffer_full,chunk_threshold)

    def addAgent(self, agent, id: str):
        self.agents[id] = agent
        self.graph.add_agent(id)
    async def addTrace(self, id: str, text: str):
        self.buffer.addchunk((id, text))

    async def _on_buffer_full(self, combined_text):
        self.graph.add_trace(id, combined_text)
        summary = await summarize(combined_text)
        self.graph.add_summary(id, summary)
