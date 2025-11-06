from typing import Optional, Union
from agents.buffer_agent import BufferAgent
from agents.summarizer_agent import SummarizerAgent
from agents.fidelity_agent import FidelityAgent
from schema.agent_summary import AgentSummary
from schema.fidelity_report import FidelityReport
import json

class EXAID:
    def __init__(self, max_summary_retries: int = 3):
        self.buffer_agent = BufferAgent()
        self.summarizer_agent = SummarizerAgent()
        self.fidelity_agent = FidelityAgent()
        self.summaries: list[AgentSummary] = []
        self.fidelity_reports: list[FidelityReport] = []
        self.max_summary_retries = max_summary_retries
    
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
        """Process a trace from an agent. Returns an AgentSummary if summarization was triggered, None otherwise.
        Includes fidelity validation with feedback loop for improvement."""
        trigger = await self.buffer_agent.addchunk(id, text)
        if trigger:
            # Capture buffer content before flushing for fidelity validation
            source_traces = self.buffer_agent.peek()
            agent_buffer = self.buffer_agent.flush()
            buffer_str = "\n".join(agent_buffer)
            
            # Get previous summaries for context
            all_summaries = self.get_all_summaries()
            summary_history_strs = self._format_summaries_history(all_summaries[:-1]) if len(all_summaries) > 1 else []
            latest_summary_str = self._format_summary_for_history(all_summaries[-1]) if all_summaries else "No summaries yet."
            
            # Generate initial summary
            summary = await self.summarizer_agent.summarize(
                summary_history_strs,
                latest_summary_str,
                buffer_str
            )
            
            # Validate summary with fidelity agent
            previous_summaries = all_summaries.copy()
            fidelity_report = await self.fidelity_agent.validate(
                summary,
                source_traces,
                previous_summaries
            )
            
            # Feedback loop: retry summarization if validation fails
            retry_count = 0
            best_summary = summary
            best_report = fidelity_report
            
            while not fidelity_report.overall_pass and retry_count < self.max_summary_retries:
                retry_count += 1
                
                # Extract issues for feedback
                feedback_issues = "\n".join(fidelity_report.issues) if fidelity_report.issues else "General fidelity issues detected."
                
                # Rerun summarizer with fidelity feedback
                summary = await self.summarizer_agent.summarize(
                    summary_history_strs,
                    latest_summary_str,
                    buffer_str,
                    fidelity_feedback=feedback_issues
                )
                
                # Re-validate
                fidelity_report = await self.fidelity_agent.validate(
                    summary,
                    source_traces,
                    previous_summaries
                )
                
                # Keep track of best summary (first passing, or update if this one has fewer issues)
                if fidelity_report.overall_pass:
                    best_summary = summary
                    best_report = fidelity_report
                    break
                else:
                    # Update best if this retry has fewer issues (simple heuristic)
                    if len(fidelity_report.issues) < len(best_report.issues):
                        best_summary = summary
                        best_report = fidelity_report
            
            # Store final summary and fidelity report
            self.summaries.append(best_summary)
            self.fidelity_reports.append(best_report)
            
            return best_summary
        return None
    
    def get_summary_json(self, summary: Optional[AgentSummary] = None) -> Optional[str]:
        """Returns a summary as JSON string. If no summary provided, returns the latest."""
        target = summary if summary else self.latest_summary()
        if isinstance(target, AgentSummary):
            return json.dumps(target.model_dump(), indent=2)
        return None
    
    def get_latest_fidelity_report(self) -> Optional[FidelityReport]:
        """Returns the latest fidelity report, or None if no reports exist."""
        if self.fidelity_reports:
            return self.fidelity_reports[-1]
        return None
    
    def get_all_fidelity_reports(self) -> list[FidelityReport]:
        """Returns all fidelity reports."""
        return self.fidelity_reports
    
    def get_fidelity_report_json(self, report: Optional[FidelityReport] = None) -> Optional[str]:
        """Returns a fidelity report as JSON string. If no report provided, returns the latest."""
        target = report if report else self.get_latest_fidelity_report()
        if isinstance(target, FidelityReport):
            return json.dumps(target.model_dump(), indent=2)
        return None
