from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ValidationCheck(BaseModel):
    """Individual validation check result."""
    passed: bool = Field(description="Whether this check passed")
    details: str = Field(description="Detailed explanation of the check result")

class FidelityReport(BaseModel):
    """Comprehensive fidelity validation report for agent summaries."""
    
    summary_to_source: Dict[str, ValidationCheck] = Field(
        description="Summary-to-source verification checks: agent_coverage, action_reasoning_capture, hallucinations, critical_info_loss"
    )
    
    completeness: Dict[str, ValidationCheck] = Field(
        description="Completeness checks: diagnostic_recommendations, treatment_plans, risk_factors, contraindications, critical_findings"
    )
    
    consistency: Dict[str, ValidationCheck] = Field(
        description="Consistency checks: conflicting_diagnoses, contradictory_recommendations, inconsistent_findings"
    )
    
    medical_accuracy: Dict[str, ValidationCheck] = Field(
        description="Medical accuracy checks: guideline_alignment, drug_interactions, dosage_accuracy, clinical_logic"
    )
    
    overall_pass: bool = Field(description="Whether all critical checks passed")
    
    issues: List[str] = Field(
        default_factory=list,
        description="List of all issues found during validation"
    )

