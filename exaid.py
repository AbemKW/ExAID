from summary_state import SummaryState
from buffer import TraceBuffer
from summarizer_agent import summarize

class EXAID:
    def __init__(self,chunk_threshold: int = 5):
        self.graph = SummaryState()
        self.agents = {}
        self.buffer = TraceBuffer(self._on_buffer_full, chunk_threshold, self.graph)

    def addAgent(self, agent, id: str):
        self.agents[id] = agent
        self.graph.add_agent(id)

    async def addTrace(self, id: str, text: str):
        # pass agent id and text separately to the buffer
        await self.buffer.addchunk(id, text)

    async def _on_buffer_full(self, agent_id: str, combined_text: str):
        # store trace and produce summary for the given agent
        self.graph.add_trace(agent_id, combined_text)

        summary, self.graph = await summarize(agent_id, combined_text, self.graph)
        self.graph.add_summary(agent_id, summary)
        
        # Process any queued items from any agent
        while not self.buffer.queue.empty():
            queued_agent_id, queued_text = self.buffer.queue.get()
            self.graph.add_trace(queued_agent_id, queued_text)
            summary, self.graph = await summarize(queued_agent_id, queued_text, self.graph)
            self.graph.add_summary(queued_agent_id, summary)

    async def getsummary(self, agent_id: str) -> str:
        summaries = self.graph.state[agent_id]["summaries"]
        if summaries:
            return summaries[-1]["text"]
        return ""
    
    async def getfullsummary(self, agent_id: str) -> list[dict]:
        return self.graph.state[agent_id]["summaries"]
