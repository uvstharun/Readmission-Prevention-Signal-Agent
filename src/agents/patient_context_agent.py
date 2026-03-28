"""
Patient Context Agent: Retrieves and summarizes full patient clinical context.
"""
from typing import Dict, List
from src.utils.logger import get_logger, log_agent_action
from src.utils.database import save_agent_decision, get_patient_risk_history

logger = get_logger(__name__)

HIGH_RISK_DIAGNOSES = {
    "I50": "Heart Failure",
    "I21": "Acute Myocardial Infarction",
    "N18": "Chronic Kidney Disease",
    "J44": "COPD",
    "A41": "Sepsis",
    "E11": "Type 2 Diabetes",
    "F32": "Depression",
    "F10": "Alcohol Use Disorder",
}

HIGH_RISK_MEDS_CLASSES = ["anticoagulant", "insulin", "opioid"]


class PatientContextAgent:
    """Extracts and summarizes relevant clinical context for a patient."""

    def run(self, patient_data: Dict) -> Dict:
        """
        Analyze patient record and return structured clinical context summary.
        """
        patient_id = patient_data.get("patient_id", "UNKNOWN")
        logger.info(f"Building context for patient {patient_id}")

        context = {
            "patient_id": patient_id,
            "demographics": self._extract_demographics(patient_data),
            "clinical_summary": self._summarize_clinical(patient_data),
            "utilization_summary": self._summarize_utilization(patient_data),
            "medication_summary": self._summarize_medications(patient_data),
            "social_context": self._summarize_social(patient_data),
            "care_transition_status": self._summarize_care_transition(patient_data),
            "risk_flags": self._identify_risk_flags(patient_data),
        }

        log_agent_action("PatientContextAgent", patient_id, "build_context",
                         f"flags={len(context['risk_flags'])}")

        save_agent_decision(
            patient_id=patient_id,
            agent_name="PatientContextAgent",
            action="build_context",
            reasoning="Extracted and structured patient clinical context from discharge record",
            output=context,
        )

        return context

    def _extract_demographics(self, d: Dict) -> Dict:
        return {
            "age": d.get("age"),
            "gender": d.get("gender"),
            "race": d.get("race"),
            "language": d.get("preferred_language"),
            "insurance": d.get("insurance_type"),
            "zip_code": d.get("zip_code"),
        }

    def _summarize_clinical(self, d: Dict) -> Dict:
        primary = d.get("primary_diagnosis_code", "")
        diag_name = "Unknown"
        for prefix, name in HIGH_RISK_DIAGNOSES.items():
            if primary.startswith(prefix):
                diag_name = name
                break

        secondary_raw = d.get("secondary_diagnosis_codes", "")
        secondary_list = str(secondary_raw).split("|") if secondary_raw and str(secondary_raw) != "nan" else []

        return {
            "primary_diagnosis_code": primary,
            "primary_diagnosis_name": diag_name,
            "secondary_diagnoses": secondary_list[:5],
            "num_secondary_diagnoses": len(secondary_list),
            "charlson_comorbidity_index": d.get("charlson_comorbidity_index", 0),
            "length_of_stay_days": d.get("length_of_stay_days"),
            "discharge_disposition": d.get("discharge_disposition"),
            "attending_department": d.get("attending_department"),
        }

    def _summarize_utilization(self, d: Dict) -> Dict:
        prior_adm = d.get("prior_admissions_6mo", 0)
        prior_ed = d.get("prior_ed_visits_6mo", 0)
        prior_readmit = d.get("prior_readmissions_1yr", 0)

        high_utilizer = prior_adm >= 2 or prior_ed >= 3

        return {
            "prior_admissions_6mo": prior_adm,
            "prior_ed_visits_6mo": prior_ed,
            "prior_readmissions_1yr": prior_readmit,
            "high_utilizer": high_utilizer,
            "utilization_severity": "HIGH" if high_utilizer else ("MODERATE" if (prior_adm >= 1 or prior_ed >= 1) else "LOW"),
        }

    def _summarize_medications(self, d: Dict) -> Dict:
        n_meds = d.get("num_active_medications", 0)
        high_risk = d.get("high_risk_medication_flag", 0)

        return {
            "num_active_medications": n_meds,
            "polypharmacy": n_meds >= 5,
            "high_risk_medications": bool(high_risk),
            "medication_risk_level": "HIGH" if high_risk else ("MODERATE" if n_meds >= 5 else "LOW"),
        }

    def _summarize_social(self, d: Dict) -> Dict:
        housing = d.get("housing_stability_flag", 1)
        transport = d.get("transportation_access_flag", 1)
        support = d.get("social_support_score", 5)
        insurance = d.get("insurance_type", "")

        sdoh_risk_count = sum([
            1 - housing,
            1 - transport,
            1 if support <= 3 else 0,
            1 if insurance in ["Medicaid", "Uninsured"] else 0,
        ])

        return {
            "housing_stable": bool(housing),
            "transportation_available": bool(transport),
            "social_support_score": support,
            "insurance_type": insurance,
            "sdoh_risk_count": sdoh_risk_count,
            "social_risk_level": "HIGH" if sdoh_risk_count >= 3 else ("MODERATE" if sdoh_risk_count >= 1 else "LOW"),
        }

    def _summarize_care_transition(self, d: Dict) -> Dict:
        followup = d.get("followup_appointment_scheduled", 0)
        pcp = d.get("pcp_assigned_flag", 0)
        instructions = d.get("discharge_instructions_given", 0)
        gaps = sum([1 - followup, 1 - pcp, 1 - instructions])

        return {
            "followup_scheduled": bool(followup),
            "pcp_assigned": bool(pcp),
            "discharge_instructions_given": bool(instructions),
            "care_transition_gaps": gaps,
            "care_transition_quality": "POOR" if gaps >= 2 else ("FAIR" if gaps == 1 else "GOOD"),
        }

    def _identify_risk_flags(self, d: Dict) -> List[str]:
        flags = []
        if d.get("charlson_comorbidity_index", 0) >= 4:
            flags.append("HIGH_COMORBIDITY_BURDEN")
        if d.get("prior_admissions_6mo", 0) >= 2:
            flags.append("FREQUENT_INPATIENT_UTILIZER")
        if d.get("prior_ed_visits_6mo", 0) >= 3:
            flags.append("FREQUENT_ED_UTILIZER")
        if d.get("high_risk_medication_flag", 0):
            flags.append("HIGH_RISK_MEDICATIONS")
        if not d.get("housing_stability_flag", 1):
            flags.append("HOUSING_INSTABILITY")
        if not d.get("followup_appointment_scheduled", 1):
            flags.append("NO_FOLLOWUP_SCHEDULED")
        if d.get("discharge_disposition") == "AMA":
            flags.append("AMA_DISCHARGE")
        if d.get("length_of_stay_days", 0) >= 7:
            flags.append("PROLONGED_HOSPITALIZATION")
        if d.get("num_active_medications", 0) >= 10:
            flags.append("POLYPHARMACY_HIGH")
        if d.get("social_support_score", 5) <= 2:
            flags.append("POOR_SOCIAL_SUPPORT")
        return flags
