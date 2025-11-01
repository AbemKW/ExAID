from typing import Optional, Union
from buffer import TraceBuffer
from agents.summarizer import summarize, AgentSummary
import json


class EXAID:
    def __init__(self):
        self.trace_buffer = TraceBuffer()
        self.summaries: list[AgentSummary] = []
    
    def latest_summary(self) -> Union[AgentSummary, str]:
        """Returns the latest summary as an AgentSummary object, or a string for initial state."""
        if self.summaries:
            return self.summaries[-1]
        return "No summaries yet."
    
    def get_all_summaries(self) -> list[AgentSummary]:
        """Returns all summaries as AgentSummary objects."""
        return self.summaries
    
    def _format_summary_for_history(self, summary: AgentSummary) -> str:
        """Converts an AgentSummary to a string representation for use in history."""
        agents_str = ", ".join(summary.agents)
        parts = [f"Agents: {agents_str}", f"Action: {summary.action}", f"Reasoning: {summary.reasoning}"]
        if summary.findings:
            parts.append(f"Findings: {summary.findings}")
        if summary.next_steps:
            parts.append(f"Next: {summary.next_steps}")
        return " | ".join(parts)
    
    def _format_summaries_history(self, summaries: list[AgentSummary]) -> list[str]:
        """Converts a list of AgentSummary objects to string representations for prompt history."""
        return [self._format_summary_for_history(s) for s in summaries]

    async def received_trace(self, id: str, text: str) -> Optional[AgentSummary]:
        """Process a trace from an agent. Returns an AgentSummary if summarization was triggered, None otherwise."""
        trigger = await self.trace_buffer.addchunk(id, text)
        if trigger:
            agent_buffer = self.trace_buffer.flush()
            buffer_str = "\n".join(agent_buffer)
            
            # Get previous summaries (excluding the latest one for comparison)
            all_summaries = self.get_all_summaries()
            summary_history_strs = self._format_summaries_history(all_summaries[:-1]) if len(all_summaries) > 1 else []
            latest_summary_str = self._format_summary_for_history(all_summaries[-1]) if all_summaries else "No summaries yet."
            
            summary = await summarize(
                summary_history_strs,
                latest_summary_str,
                buffer_str
            )
            self.summaries.append(summary)
            return summary
        return None
    
    def get_summary_json(self, summary: Optional[AgentSummary] = None) -> Optional[str]:
        """Returns a summary as JSON string. If no summary provided, returns the latest."""
        target = summary if summary else self.latest_summary()
        if isinstance(target, AgentSummary):
            return json.dumps(target.model_dump(), indent=2)
        return None
