import sys
from pathlib import Path

# CRITICAL: Add parent directory to path BEFORE any other imports
# This must happen first so that exaid.py can find its dependencies
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can safely import modules that depend on the project root being in sys.path
from typing import Union
from exaid import EXAID
from schema.agent_summary import AgentSummary
from cdss_demo.agents.orchestrator_agent import OrchestratorAgent
from cdss_demo.agents.cardiology_agent import CardiologyAgent
from cdss_demo.agents.laboratory_agent import LaboratoryAgent
from cdss_demo.schema.clinical_case import ClinicalCase


class CDSS:
    """Clinical Decision Support System orchestrator"""
    
    def __init__(self):
        """Initialize CDSS with EXAID and specialized agents"""
        self.exaid = EXAID()
        self.orchestrator = OrchestratorAgent()
        self.cardiology = CardiologyAgent()
        self.laboratory = LaboratoryAgent()
    
    async def process_case(
        self, 
        case: Union[ClinicalCase, str],
        use_streaming: bool = True
    ) -> dict:
        """Process a clinical case through the multi-agent system
        
        Args:
            case: ClinicalCase object or free-text case description
            use_streaming: Whether to use streaming token processing (default: True)
            
        Returns:
            Dictionary containing agent findings and final recommendation
        """
        # Convert case to clinical summary if it's a ClinicalCase object
        if isinstance(case, ClinicalCase):
            case_text = case.to_clinical_summary()
        else:
            case_text = str(case)
        
        # Step 1: Orchestrator analyzes case and determines workflow
        orchestrator_input = (
            f"Clinical Case:\n{case_text}\n\n"
            "Analyze this case and determine which specialist agents should be consulted. "
            "Identify key clinical questions that need to be answered."
        )
        
        if use_streaming:
            token_stream = self.orchestrator.act_stream(orchestrator_input)
            await self.exaid.received_streamed_tokens(
                self.orchestrator.agent_id,
                token_stream
            )
        else:
            orchestrator_trace = await self.orchestrator.act(orchestrator_input)
            await self.exaid.received_trace(
                self.orchestrator.agent_id,
                orchestrator_trace
            )
        
        # Step 2: Laboratory agent analyzes lab results
        lab_input = (
            f"Clinical Case:\n{case_text}\n\n"
            "Analyze the laboratory results and provide interpretation. "
            "Identify any abnormal values, critical findings, or patterns that suggest specific diagnoses. "
            "Recommend additional tests if needed."
        )
        
        if use_streaming:
            token_stream = self.laboratory.act_stream(lab_input)
            await self.exaid.received_streamed_tokens(
                self.laboratory.agent_id,
                token_stream
            )
        else:
            lab_trace = await self.laboratory.act(lab_input)
            await self.exaid.received_trace(
                self.laboratory.agent_id,
                lab_trace
            )
        
        # Step 3: Cardiology agent assesses cardiac aspects
        cardio_input = (
            f"Clinical Case:\n{case_text}\n\n"
            "Assess the cardiac aspects of this case. Consider:\n"
            "- Cardiovascular risk factors\n"
            "- Cardiac symptoms and signs\n"
            "- Cardiac biomarkers and tests\n"
            "- ECG or imaging findings if available\n"
            "- Cardiac medication considerations\n"
            "Provide cardiac assessment and recommendations."
        )
        
        if use_streaming:
            token_stream = self.cardiology.act_stream(cardio_input)
            await self.exaid.received_streamed_tokens(
                self.cardiology.agent_id,
                token_stream
            )
        else:
            cardio_trace = await self.cardiology.act(cardio_input)
            await self.exaid.received_trace(
                self.cardiology.agent_id,
                cardio_trace
            )
        
        # Step 4: Orchestrator synthesizes findings
        # Get all summaries for context
        all_summaries = self.exaid.get_all_summaries()
        summary_context = "\n\n".join([
            f"Agent: {', '.join(s.agents)}\n"
            f"Action: {s.action}\n"
            f"Reasoning: {s.reasoning}\n"
            f"Findings: {s.findings or 'N/A'}\n"
            f"Next Steps: {s.next_steps or 'N/A'}"
            for s in all_summaries
        ])
        
        synthesis_input = (
            f"Original Clinical Case:\n{case_text}\n\n"
            f"Agent Findings and Summaries:\n{summary_context}\n\n"
            "Synthesize all findings from the specialist agents into a comprehensive "
            "clinical assessment and recommendation. Provide:\n"
            "- Overall clinical assessment\n"
            "- Key findings from each specialist\n"
            "- Integrated diagnosis or differential diagnosis\n"
            "- Prioritized recommendations\n"
            "- Follow-up plan"
        )
        
        if use_streaming:
            token_stream = self.orchestrator.act_stream(synthesis_input)
            final_summary = await self.exaid.received_streamed_tokens(
                self.orchestrator.agent_id,
                token_stream
            )
        else:
            final_trace = await self.orchestrator.act(synthesis_input)
            final_summary = await self.exaid.received_trace(
                self.orchestrator.agent_id,
                final_trace
            )
        
        all_summaries = self.exaid.get_all_summaries()
        
        # Compile results
        result = {
            "case_summary": case_text,
            "agent_summaries": [
                {
                    "agents": s.agents,
                    "action": s.action,
                    "reasoning": s.reasoning,
                    "findings": s.findings,
                    "next_steps": s.next_steps
                }
                for s in all_summaries
            ],
            "final_recommendation": {
                "agents": final_summary.agents if final_summary else [],
                "action": final_summary.action if final_summary else "",
                "reasoning": final_summary.reasoning if final_summary else "",
                "findings": final_summary.findings if final_summary else None,
                "next_steps": final_summary.next_steps if final_summary else None
            },
            "trace_count": {
                "orchestrator": self.exaid.get_agent_trace_count(self.orchestrator.agent_id),
                "cardiology": self.exaid.get_agent_trace_count(self.cardiology.agent_id),
                "laboratory": self.exaid.get_agent_trace_count(self.laboratory.agent_id)
            }
        }
        
        return result
    
    def get_all_summaries(self) -> list[AgentSummary]:
        """Get all summaries from EXAID"""
        return self.exaid.get_all_summaries()
    
    def get_summaries_by_agent(self, agent_id: str) -> list[AgentSummary]:
        """Get summaries for a specific agent"""
        return self.exaid.get_summaries_by_agent(agent_id)
    
    def reset(self):
        """Reset the CDSS system (create new EXAID instance)"""
        self.exaid = EXAID()

