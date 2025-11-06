from langchain_core.prompts import ChatPromptTemplate
from schema.fidelity_report import FidelityReport
from schema.agent_summary import AgentSummary
from typing import List
from llm import llm


class FidelityAgent:
    """Validates agent summaries for fidelity, completeness, consistency, and medical accuracy."""
    
    def __init__(self):
        self.llm = llm.with_structured_output(schema=FidelityReport)
        self.validation_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a fidelity validation agent for a medical multi-agent reasoning system. "
             "Your task is to validate summaries against source traces and check for completeness, consistency, and medical accuracy. "
             "Be thorough and critical - patient safety depends on accurate summaries.\n\n"
             
             "VALIDATION DIMENSIONS:\n\n"
             
             "1. SUMMARY-TO-SOURCE VERIFICATION (summary_to_source dict):\n"
             "Provide checks with these exact keys:\n"
             "- agent_coverage: All agents mentioned in traces must be in summary.agents\n"
             "- action_reasoning_capture: Key actions/reasoning from traces must be captured in summary\n"
             "- hallucinations: Summary must not contain information not present in source traces\n"
             "- critical_info_loss: Critical information from traces must not be lost in summary\n\n"
             
             "2. COMPLETENESS CHECKING (completeness dict):\n"
             "For medical context, ensure summaries capture (provide checks with these exact keys):\n"
             "- diagnostic_recommendations: Any diagnostic recommendations mentioned\n"
             "- treatment_plans: Treatment plans or interventions discussed\n"
             "- risk_factors: Risk factors identified\n"
             "- contraindications: Contraindications or warnings mentioned\n"
             "- critical_findings: Critical clinical findings\n\n"
             
             "3. CONSISTENCY VALIDATION (consistency dict):\n"
             "Check for consistency across summaries (provide checks with these exact keys):\n"
             "- conflicting_diagnoses: No conflicting diagnoses between summaries\n"
             "- contradictory_recommendations: No contradictory treatment recommendations\n"
             "- inconsistent_findings: Findings should be consistent across summaries\n\n"
             
             "4. MEDICAL ACCURACY VERIFICATION (medical_accuracy dict):\n"
             "Validate medical claims (provide checks with these exact keys):\n"
             "- guideline_alignment: Treatment recommendations align with medical guidelines (NCCN, etc.)\n"
             "- drug_interactions: Check for drug interactions or contraindications\n"
             "- dosage_accuracy: Verify dosage information is reasonable\n"
             "- clinical_logic: Clinical reasoning should be logically consistent\n\n"
             
             "For each check, provide:\n"
             "- passed: true if check passes, false if issues found\n"
             "- details: Specific explanation of what was checked and any issues found\n\n"
             
             "Set overall_pass to false if ANY critical check fails. "
             "Collect all issues into the issues list as strings for feedback."),
            
            ("user",
             "Source traces:\n{source_traces}\n\n"
             "Summary to validate:\n{summary}\n\n"
             "Previous summaries (for consistency checking):\n{previous_summaries}\n\n"
             "Validate this summary across all dimensions and provide a comprehensive fidelity report.")
        ])
    
    def _format_summary(self, summary: AgentSummary) -> str:
        """Format AgentSummary for prompt."""
        agents_str = ", ".join(summary.agents)
        parts = [
            f"Agents: {agents_str}",
            f"Action: {summary.action}",
            f"Reasoning: {summary.reasoning}"
        ]
        if summary.findings:
            parts.append(f"Findings: {summary.findings}")
        if summary.next_steps:
            parts.append(f"Next Steps: {summary.next_steps}")
        return "\n".join(parts)
    
    def _format_previous_summaries(self, summaries: List[AgentSummary]) -> str:
        """Format list of summaries for prompt."""
        if not summaries:
            return "No previous summaries."
        return "\n---\n".join([self._format_summary(s) for s in summaries])
    
    async def validate(
        self,
        summary: AgentSummary,
        source_traces: List[str],
        previous_summaries: List[AgentSummary]
    ) -> FidelityReport:
        """Validate a summary against source traces and previous summaries.
        
        Args:
            summary: The AgentSummary to validate
            source_traces: List of original trace strings from the buffer
            previous_summaries: List of previous AgentSummary objects for consistency checking
            
        Returns:
            FidelityReport with validation results
        """
        validation_chain = self.validation_prompt | self.llm
        
        source_traces_str = "\n".join(source_traces)
        summary_str = self._format_summary(summary)
        previous_summaries_str = self._format_previous_summaries(previous_summaries)
        
        report = await validation_chain.ainvoke({
            "source_traces": source_traces_str,
            "summary": summary_str,
            "previous_summaries": previous_summaries_str
        })
        
        return report

