"""
All LLM prompt templates for the readmission prevention agent.
"""

SYSTEM_CLINICAL_NARRATOR = """You are a senior clinical informatics specialist and case management expert.
You analyze patient discharge data and readmission risk scores to generate clear, actionable clinical narratives
for care transition teams. Your narratives are evidence-based, clinically precise, and written for an audience
of case managers, discharge planners, and attending physicians.

Always structure your response as valid JSON with the exact keys requested. Be concise and clinically specific."""

CLINICAL_NARRATIVE_PROMPT = """Analyze this patient's readmission risk profile and generate a structured clinical narrative.

PATIENT PROFILE:
- Patient ID: {patient_id}
- Age: {age} | Gender: {gender}
- Primary Diagnosis: {primary_diagnosis} ({primary_ccsr})
- Length of Stay: {los} days
- Discharge Disposition: {discharge_disposition}
- Charlson Comorbidity Index: {cci}

RISK ASSESSMENT:
- 30-Day Readmission Risk Score: {risk_score:.1%}
- Risk Tier: {risk_tier}

TOP RISK DRIVERS (SHAP Analysis):
{top_risk_drivers}

CARE GAPS IDENTIFIED:
{care_gaps}

SOCIAL DETERMINANTS:
- Housing Stability: {housing}
- Transportation Access: {transport}
- Social Support Score: {social_support}/7
- Insurance: {insurance}

UTILIZATION HISTORY:
- Prior admissions (6 months): {prior_admissions}
- Prior ED visits (6 months): {prior_ed}
- Prior readmissions (1 year): {prior_readmissions}

CARE TRANSITION STATUS:
- Follow-up appointment scheduled: {followup}
- PCP assigned: {pcp_assigned}
- Discharge instructions given: {dc_instructions}

Generate a clinical narrative as JSON with these exact keys:
{{
  "risk_summary": "One sentence summarizing overall risk and primary driver",
  "clinical_rationale": "2-3 sentences explaining the clinical basis for this risk score, referencing specific data points",
  "top_risk_drivers_explained": [
    {{"driver": "factor name", "explanation": "plain clinical language explanation of why this increases risk"}}
  ],
  "intervention_tier_rationale": "1-2 sentences explaining why this tier and intervention intensity is appropriate",
  "case_manager_talking_points": [
    "Specific talking point 1 for the care team",
    "Specific talking point 2",
    "Specific talking point 3"
  ],
  "priority_actions": [
    "Immediate action 1",
    "Immediate action 2"
  ]
}}"""

CARE_GAP_ANALYSIS_PROMPT = """Review this patient's care transition status and identify evidence-based care gaps.

Patient: {patient_id} | Age: {age} | Discharge: {discharge_date}
Primary Dx: {primary_diagnosis} | Risk Score: {risk_score:.1%} | Tier: {risk_tier}

Current Care Transition Status:
- Follow-up appointment scheduled: {followup}
- PCP assigned: {pcp_assigned}
- Discharge instructions given: {dc_instructions}
- Number of active medications: {num_meds}
- High-risk medications (anticoagulants/insulin/opioids): {high_risk_meds}
- Housing stability: {housing}
- Transportation access: {transport}
- Social support score: {social_support}/7

Based on evidence-based care transition guidelines (BOOST, Project RED, CARE Transitions),
identify the top 3-5 most critical care gaps as JSON:
{{
  "care_gaps": [
    {{
      "gap": "gap name",
      "priority": "HIGH/MEDIUM/LOW",
      "rationale": "clinical rationale",
      "recommended_intervention": "specific action"
    }}
  ],
  "overall_care_gap_severity": "HIGH/MEDIUM/LOW",
  "recommended_bundle": "brief description of recommended intervention bundle"
}}"""

MONITORING_ESCALATION_PROMPT = """A monitored post-discharge patient has triggered escalation signals. Assess whether clinical escalation is warranted.

Patient: {patient_id} | Days Post-Discharge: {days_post_discharge}
Original Risk Tier: {original_tier} | Original Score: {original_score:.1%}

NEW SIGNALS DETECTED:
{signals}

Current Watchlist Status: {watchlist_status}

Based on these signals, provide escalation assessment as JSON:
{{
  "escalate": true/false,
  "escalation_urgency": "IMMEDIATE/WITHIN_24H/WITHIN_48H/MONITOR",
  "clinical_rationale": "explanation",
  "recommended_actions": ["action1", "action2"]
}}"""
