import asyncio
from exaid import EXAID

def format_summary_display(summary) -> str:
    """Format an AgentSummary for clean console display optimized for physician review."""
    agents_str = ", ".join(summary.agents)
    lines = [
        f"┌─ Agents: {agents_str}",
        f"├─ Action: {summary.action}",
        f"├─ Reasoning: {summary.reasoning}",
    ]
    if summary.findings:
        lines.append(f"├─ Findings: {summary.findings}")
    if summary.next_steps:
        lines.append(f"└─ Next Steps: {summary.next_steps}")
    else:
        # Replace last ├ with └ if no next_steps
        lines[-1] = lines[-1].replace("├─", "└─")
    return "\n".join(lines)

async def main():
    exaid = EXAID()
    # Simulate traces from two agents
    traces = [
        ("Orchestrator", "Patient case received: 62F, 6-week history of fatigue, 10-lb weight loss, night sweats, intermittent low-grade fevers. No recent travel or sick contacts. Past history: rheumatoid arthritis (on methotrexate), hypothyroidism. Initiating diagnostic workflow."),
        
        ("Orchestrator", "Activating InfectiousDiseaseAgent, HematologyAgent, and AutoimmuneAgent for initial hypothesis generation."),
        
        ("InfectiousDiseaseAgent", "Considering subacute infections: tuberculosis (given night sweats and weight loss), endocarditis (though no murmur reported), and occult abscess. Risk factors include immunosuppression from methotrexate. Recommend CXR, blood cultures, and Quantiferon-TB Gold."),
        
        ("HematologyAgent", "Constitutional B symptoms (fever, night sweats, weight loss) raise concern for lymphoproliferative disorders—particularly Hodgkin or non-Hodgkin lymphoma. Also considering chronic leukemia. Recommend CBC with differential, LDH, peripheral smear, and consider CT chest/abdomen/pelvis for lymphadenopathy."),
        
        ("AutoimmuneAgent", "Methotrexate can rarely cause drug-induced hypersensitivity syndrome, but more likely: underlying rheumatoid arthritis flare or secondary amyloidosis. However, fever pattern is atypical for pure RA flare. Suggest CRP, ESR, ferritin, and review for serositis."),
        
        ("Orchestrator", "Initial labs show: Hb 9.8 g/dL (normocytic), WBC 14.2k/µL (neutrophilia), ESR 88 mm/hr, CRP 12 mg/dL, LDH 320 U/L. CXR reveals right hilar fullness. Activating RadiologyAgent, OncologyAgent, and InflammationAgent."),
        
        ("RadiologyAgent", "Chest X-ray shows right hilar prominence with mediastinal widening. No infiltrates or effusions. Highly suspicious for mediastinal lymphadenopathy. Recommend contrast-enhanced CT chest without delay."),
        
        ("OncologyAgent", "Hilar/mediastinal lymphadenopathy + B symptoms strongly suggest lymphoma. Age and immunosuppression increase risk. Next step: CT-guided biopsy or excisional lymph node biopsy if peripheral nodes accessible. Also consider PET-CT for staging if diagnosis confirmed."),
        
        ("InflammationAgent", "Markedly elevated ESR and CRP indicate systemic inflammation. Ferritin not yet checked—important to rule out adult-onset Still's disease or hemophagocytic lymphohistiocytosis (HLH), though less likely without cytopenias or hepatosplenomegaly. Recommend ferritin, fibrinogen, triglycerides."),
        
        ("Orchestrator", "CT chest shows bulky mediastinal and right hilar lymphadenopathy, largest node 3.2 cm. No hepatosplenomegaly. Ferritin: 420 ng/mL (normal). LDH elevated. Activating PathologyAgent and ImmunologyAgent."),
        
        ("PathologyAgent", "Awaiting biopsy. If tissue obtained, will assess for Reed-Sternberg cells (Hodgkin), immunophenotyping (CD markers), and molecular studies (e.g., BCL2, MYC). Preliminary impression based on imaging and symptoms: classical Hodgkin lymphoma, nodular sclerosis subtype most likely in this demographic."),
        
        ("ImmunologyAgent", "Methotrexate-associated lymphoproliferative disorder (MTX-LPD) is a key differential in RA patients on long-term immunosuppression. Often EBV-driven. If biopsy is EBV+, consider MTX withdrawal as first step—spontaneous regression occurs in ~30% of cases."),
        
        ("Orchestrator", "Biopsy results: CD30+, CD15+, PAX5+, EBV-negative. Diagnosis: classical Hodgkin lymphoma, nodular sclerosis type. Activating TreatmentPlanningAgent, PharmacyAgent, and SupportiveCareAgent."),
        
        ("TreatmentPlanningAgent", "Per NCCN guidelines: Stage II (mediastinal involvement) Hodgkin lymphoma. Recommend ABVD regimen (doxorubicin, bleomycin, vinblastine, dacarbazine) × 4–6 cycles. Bleomycin caution due to age—monitor pulmonary function. Consider interim PET-CT after 2 cycles for response assessment."),
        
        ("PharmacyAgent", "ABVD dosing: doxorubicin 25 mg/m² IV, bleomycin 10 U/m² IV, vinblastine 6 mg/m² IV, dacarbazine 375 mg/m² IV on days 1 and 15. Verify GFR, LVEF baseline, and PFTs before bleomycin. Methotrexate should be held during chemotherapy."),
        
        ("SupportiveCareAgent", "Anticipate chemotherapy side effects: myelosuppression (monitor CBC weekly), nausea (ondansetron PRN), fatigue. Provide fertility counseling (patient is perimenopausal). Screen for depression given new cancer diagnosis. Arrange social work and nutrition consult."),
        
        ("Orchestrator", "Final diagnostic consensus: Classical Hodgkin lymphoma, Stage IIA. Treatment plan aligned with NCCN, adjusted for comorbidities and medication history. All agents concur. Proceed to oncology referral and cycle 1 scheduling.")
    ]

    for agent_id, text in traces:
        summary = await exaid.received_trace(agent_id, text)
        if summary:
            print(f"\n{'='*60}")
            print(f"Summary Update")
            print(f"{'='*60}")
            print(format_summary_display(summary))
            print()

if __name__ == "__main__":
    asyncio.run(main())
