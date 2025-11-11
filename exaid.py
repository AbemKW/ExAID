from typing import Optional, Union, AsyncIterator
from agents.buffer_agent import BufferAgent
from agents.summarizer_agent import SummarizerAgent
from agents.token_gate import TokenGate
from schema.agent_summary import AgentSummary
import json
import asyncio

class EXAID:
    def __init__(self):
        self.buffer_agent = BufferAgent()
        self.summarizer_agent = SummarizerAgent()
        self.token_gate = TokenGate()
        self.summaries: list[AgentSummary] = []
    
    def latest_summary(self) -> Union[AgentSummary, str]:
        """Returns the latest summary as an AgentSummary object, or a string for initial state."""
        if self.summaries:
            return self.summaries[-1]
        return "No summaries yet."
    
    def get_all_summaries(self) -> list[AgentSummary]:
        """Returns all summaries as AgentSummary objects."""
        return self.summaries

    def get_summaries_by_agent(self, agent_id: str) -> list[AgentSummary]:
        """Get all summaries involving a specific agent."""
        return [s for s in self.summaries if agent_id in s.agents]

    def get_agent_trace_count(self, agent_id: str) -> int:
        return self.buffer_agent.get_trace_count(agent_id)

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
        trigger = await self.buffer_agent.addchunk(id, text)
        if trigger:
            agent_buffer = self.buffer_agent.flush()
            buffer_str = "\n".join(agent_buffer)
            
            # Get previous summaries for context
            all_summaries = self.get_all_summaries()
            summary_history_strs = self._format_summaries_history(all_summaries[:-1]) if len(all_summaries) > 1 else []
            latest_summary_str = self._format_summary_for_history(all_summaries[-1]) if all_summaries else "No summaries yet."
            
            # Generate summary
            summary = await self.summarizer_agent.summarize(
                summary_history_strs,
                latest_summary_str,
                buffer_str
            )
            
            # Store summary
            self.summaries.append(summary)
            
            return summary
        return None
    
    async def received_streamed_tokens(self, agent_id: str, token_generator: AsyncIterator[str]) -> Optional[AgentSummary]:
        """Process streamed tokens from a long reasoning stream.
        
        Tokens flow through TokenGate (structural pre-buffer) before reaching BufferAgent
        (semantic completeness evaluator). TokenGate regulates flow based on structural
        cues without interpreting meaning.
        
        Args:
            agent_id: Identifier for the agent sending the stream
            token_generator: Async iterator yielding tokens as strings
            
        Returns:
            AgentSummary if summarization was triggered during stream processing, None otherwise
        """
        last_summary = None
        
        # Helper function to process a chunk from TokenGate through BufferAgent
        async def process_chunk(chunk: str) -> Optional[AgentSummary]:
            """Process a chunk from TokenGate through BufferAgent and generate summary if triggered."""
            trigger = await self.buffer_agent.addchunk(agent_id, chunk)
            if trigger:
                agent_buffer = self.buffer_agent.flush()
                buffer_str = "\n".join(agent_buffer)
                
                # Get previous summaries for context
                all_summaries = self.get_all_summaries()
                summary_history_strs = self._format_summaries_history(all_summaries[:-1]) if len(all_summaries) > 1 else []
                latest_summary_str = self._format_summary_for_history(all_summaries[-1]) if all_summaries else "No summaries yet."
                
                # Generate summary
                summary = await self.summarizer_agent.summarize(
                    summary_history_strs,
                    latest_summary_str,
                    buffer_str
                )
                
                # Store summary
                self.summaries.append(summary)
                return summary
            return None
        
        # Process each token in the stream through TokenGate
        async for token in token_generator:
            # Add token to TokenGate - it returns a chunk if flush conditions are met
            chunk = await self.token_gate.add_token(agent_id, token)
            
            if chunk:
                # TokenGate decided to flush - pass chunk to BufferAgent
                summary = await process_chunk(chunk)
                if summary:
                    last_summary = summary
            
            # Check if timers expired (silence timer or max wait timeout)
            timer_chunk = await self.token_gate.check_timers(agent_id)
            if timer_chunk:
                summary = await process_chunk(timer_chunk)
                if summary:
                    last_summary = summary
        
        # Flush any remaining content at stream end
        remaining = await self.token_gate.flush(agent_id)
        if remaining:
            summary = await process_chunk(remaining)
            if summary:
                last_summary = summary
        
        return last_summary
    
    def get_summary_json(self, summary: Optional[AgentSummary] = None) -> Optional[str]:
        """Returns a summary as JSON string. If no summary provided, returns the latest."""
        target = summary if summary else self.latest_summary()
        if isinstance(target, AgentSummary):
            return json.dumps(target.model_dump(), indent=2)
        return None
