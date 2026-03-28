"""
Clinical narrative generator using Claude LLM.
Transforms structured patient data into actionable clinical narratives.
"""
import json
from typing import Dict, List, Optional
from src.llm.claude_client import ClaudeClient
from src.llm.prompt_templates import (
    SYSTEM_CLINICAL_NARRATOR,
    CLINICAL_NARRATIVE_PROMPT,
    CARE_GAP_ANALYSIS_PROMPT,
    MONITORING_ESCALATION_PROMPT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClinicalNarrator:
    """Generates LLM-powered clinical narratives for readmission risk."""

    def __init__(self, client: Optional[ClaudeClient] = None):
        self.client = client or ClaudeClient()

    def _format_risk_drivers(self, top_features: List[Dict]) -> str:
        """Format SHAP features for prompt."""
        lines = []
        for i, f in enumerate(top_features[:5], 1):
            direction = f.get("direction", "increases")
            contribution = abs(f.get("shap_contribution", 0))
            lines.append(
                f"{i}. {f['feature']}: value={f['value']:.2f}, "
                f"{direction} risk (SHAP={contribution:.3f})"
            )
        return "\n".join(lines)

    def _format_care_gaps(self, care_gaps: List[Dict]) -> str:
        """Format care gaps for prompt."""
        if not care_gaps:
            return "No care gaps identified"
        lines = []
        for gap in care_gaps[:5]:
            lines.append(f"- {gap.get('gap', '')}: {gap.get('rationale', '')}")
        return "\n".join(lines)

    def generate_narrative(self, patient_data: Dict, risk_result: Dict, care_gaps: List[Dict] = None) -> Dict:
        """
        Generate full clinical narrative for a patient.

        Args:
            patient_data: Raw patient record
            risk_result: Output from risk scoring agent (score, tier, top_risk_drivers)
            care_gaps: List of identified care gaps

        Returns:
            Structured clinical narrative dict
        """
        care_gaps = care_gaps or []

        prompt = CLINICAL_NARRATIVE_PROMPT.format(
            patient_id=patient_data.get("patient_id", "UNKNOWN"),
            age=patient_data.get("age", "N/A"),
            gender=patient_data.get("gender", "N/A"),
            primary_diagnosis=patient_data.get("primary_diagnosis_code", "N/A"),
            primary_ccsr=patient_data.get("primary_ccsr_category", "N/A"),
            los=patient_data.get("length_of_stay_days", "N/A"),
            discharge_disposition=patient_data.get("discharge_disposition", "N/A"),
            cci=patient_data.get("charlson_comorbidity_index", "N/A"),
            risk_score=risk_result.get("risk_score", 0),
            risk_tier=risk_result.get("risk_tier", "N/A"),
            top_risk_drivers=self._format_risk_drivers(risk_result.get("top_risk_drivers", [])),
            care_gaps=self._format_care_gaps(care_gaps),
            housing="Stable" if patient_data.get("housing_stability_flag", 1) else "Unstable",
            transport="Available" if patient_data.get("transportation_access_flag", 1) else "Limited",
            social_support=patient_data.get("social_support_score", "N/A"),
            insurance=patient_data.get("insurance_type", "N/A"),
            prior_admissions=patient_data.get("prior_admissions_6mo", 0),
            prior_ed=patient_data.get("prior_ed_visits_6mo", 0),
            prior_readmissions=patient_data.get("prior_readmissions_1yr", 0),
            followup="Yes" if patient_data.get("followup_appointment_scheduled", 0) else "No",
            pcp_assigned="Yes" if patient_data.get("pcp_assigned_flag", 0) else "No",
            dc_instructions="Yes" if patient_data.get("discharge_instructions_given", 0) else "No",
        )

        try:
            response_text = self.client.complete(
                prompt=prompt,
                system=SYSTEM_CLINICAL_NARRATOR,
                max_tokens=1500,
                temperature=0.3,
            )

            # Parse JSON response
            # Find JSON block in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                narrative = json.loads(response_text[start:end])
            else:
                raise ValueError("No JSON found in response")

            logger.info(f"Generated narrative for patient {patient_data.get('patient_id')}")
            return narrative

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse narrative JSON: {e}")
            return self._fallback_narrative(patient_data, risk_result)

        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            return self._fallback_narrative(patient_data, risk_result)

    def _fallback_narrative(self, patient_data: Dict, risk_result: Dict) -> Dict:
        """Fallback narrative when LLM is unavailable."""
        score = risk_result.get("risk_score", 0)
        tier = risk_result.get("risk_tier", "UNKNOWN")
        return {
            "risk_summary": f"Patient has {tier} readmission risk with score of {score:.1%}",
            "clinical_rationale": "Risk assessment based on clinical and utilization data.",
            "top_risk_drivers_explained": [
                {"driver": f["feature"], "explanation": f"Contributes to increased readmission risk"}
                for f in risk_result.get("top_risk_drivers", [])[:3]
            ],
            "intervention_tier_rationale": f"Based on {tier} risk tier, standard care transition protocol applies.",
            "case_manager_talking_points": [
                "Review medication reconciliation",
                "Confirm follow-up appointment",
                "Assess social support needs",
            ],
            "priority_actions": ["Schedule follow-up within 7 days", "Complete medication reconciliation"],
        }

    def generate_care_gap_analysis(self, patient_data: Dict, risk_result: Dict) -> Dict:
        """Generate LLM-based care gap analysis."""
        prompt = CARE_GAP_ANALYSIS_PROMPT.format(
            patient_id=patient_data.get("patient_id"),
            age=patient_data.get("age"),
            discharge_date=patient_data.get("discharge_date", "N/A"),
            primary_diagnosis=patient_data.get("primary_diagnosis_code", "N/A"),
            risk_score=risk_result.get("risk_score", 0),
            risk_tier=risk_result.get("risk_tier", "N/A"),
            followup="Yes" if patient_data.get("followup_appointment_scheduled", 0) else "No",
            pcp_assigned="Yes" if patient_data.get("pcp_assigned_flag", 0) else "No",
            dc_instructions="Yes" if patient_data.get("discharge_instructions_given", 0) else "No",
            num_meds=patient_data.get("num_active_medications", 0),
            high_risk_meds="Yes" if patient_data.get("high_risk_medication_flag", 0) else "No",
            housing="Stable" if patient_data.get("housing_stability_flag", 1) else "Unstable",
            transport="Available" if patient_data.get("transportation_access_flag", 1) else "Limited",
            social_support=patient_data.get("social_support_score", "N/A"),
        )

        try:
            response_text = self.client.complete(prompt=prompt, system=SYSTEM_CLINICAL_NARRATOR, max_tokens=800)
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response_text[start:end])
        except Exception as e:
            logger.error(f"Care gap analysis failed: {e}")

        return {
            "care_gaps": [],
            "overall_care_gap_severity": "UNKNOWN",
            "recommended_bundle": "Standard care transition bundle",
        }
