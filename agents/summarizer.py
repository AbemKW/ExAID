from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional
from llm import llm


class AgentSummary(BaseModel):
    """Structured summary for medical multi-agent reasoning, optimized for physician understanding."""
    agents: List[str] = Field(description="List of agent IDs involved in this reasoning step")
    action: str = Field(max_length=100, description="Brief action statement describing what the agents did")
    reasoning: str = Field(max_length=200, description="Concise reasoning explaining why this action was taken")
    findings: Optional[str] = Field(max_length=150, default=None, description="Key clinical findings or recommendations if applicable")
    next_steps: Optional[str] = Field(max_length=100, default=None, description="Suggested next actions if applicable")
    
    @model_validator(mode='before')
    @classmethod
    def truncate_fields(cls, data):
        """Truncate fields to meet length constraints if they exceed limits."""
        if isinstance(data, dict):
            if 'action' in data and len(data.get('action', '')) > 100:
                data['action'] = data['action'][:97] + '...'
            if 'reasoning' in data and len(data.get('reasoning', '')) > 200:
                data['reasoning'] = data['reasoning'][:197] + '...'
            if 'findings' in data and data.get('findings') and len(data['findings']) > 150:
                data['findings'] = data['findings'][:147] + '...'
            if 'next_steps' in data and data.get('next_steps') and len(data['next_steps']) > 100:
                data['next_steps'] = data['next_steps'][:97] + '...'
        return data


# Create structured LLM with the schema
structured_llm = llm.with_structured_output(schema=AgentSummary)

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarizer for a medical multi-agent reasoning system. "
    "Extract essential information about agent actions and clinical reasoning from the new buffer. "
    "Focus on what agents did and why, using brief, action-oriented language. "
    "Be concise and practical - physicians need to quickly understand agent decisions. "
    "Extract only new information from the buffer, do not repeat previous summaries. "
    "Identify all agents mentioned in the buffer. "
    "CRITICAL: Strictly enforce character limits - action: MAX 100 chars, reasoning: MAX 200 chars, findings: MAX 150 chars, next_steps: MAX 100 chars. "
    "If content exceeds limits, prioritize the most essential information and truncate."),
    ("user", "Summary history:\n[ {summary_history} ]\n\nLatest summary:\n{latest_summary}\n\nNew reasoning buffer:\n{new_buffer}\n\nExtract structured summary of new agent actions and reasoning."),
]
)

async def summarize(summary_history: List[str], latest_summary: str, new_buffer: str) -> AgentSummary:
    """Updates the summary given the summary history (as a list), latest summary, and new reasoning buffer.
    Returns a structured AgentSummary object."""
    summarize_chain = summarize_prompt | structured_llm
    summary = await summarize_chain.ainvoke({
        "summary_history": ",\n".join(summary_history),
        "latest_summary": latest_summary,
        "new_buffer": new_buffer
    })
    return summary