from typing import Optional
from buffer import TraceBuffer
from agents.summarizer import summarize

class EXAID:
    def __init__(self):
        self.trace_buffer = TraceBuffer()
        self.summaries = []
    
    def latest_summary(self):
        if self.summaries:
            return self.summaries[-1]
        return "No summaries yet."
    def get_all_summaries(self):
        if self.summaries:
            return self.summaries

    async def received_trace(self, id: str, text: str) -> Optional[str]:
        # pass agent id and text separately to the buffer
        trigger = await self.trace_buffer.addchunk(id, text)
        if trigger:
            agent_buffer = self.trace_buffer.flush(id)
            buffer_str = "\n".join(agent_buffer)
            summary = await summarize(
                self.get_all_summaries()[:-1],
                self.latest_summary(),
                buffer_str
            )
            self.summaries.append(summary)
            return summary
        return None
