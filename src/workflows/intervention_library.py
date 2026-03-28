"""
Library of evidence-based care transition interventions.
Each intervention has trigger conditions, responsible team, priority, and evidence base.
"""
from typing import Dict, List

INTERVENTION_LIBRARY = [
    {
        "id": "INT001",
        "name": "Medication Reconciliation Review",
        "type": "clinical",
        "responsible_team": "Clinical Pharmacist",
        "priority": "HIGH",
        "evidence_base": "JCAHO National Patient Safety Goal 03.06.01; reduces ADEs by 50%",
        "trigger_conditions": ["polypharmacy", "high_risk_medication", "any_high_risk"],
        "estimated_duration_minutes": 30,
        "template_message": (
            "MEDICATION RECONCILIATION REQUIRED: Patient {patient_id} has {n_meds} active medications "
            "at discharge. Please complete medication reconciliation within 24 hours, with particular "
            "attention to high-risk medications."
        ),
    },
    {
        "id": "INT002",
        "name": "PCP Follow-Up Appointment (7 Days)",
        "type": "care_coordination",
        "responsible_team": "Care Coordinator",
        "priority": "HIGH",
        "evidence_base": "Project RED; BOOST; 7-day follow-up reduces readmission by 20%",
        "trigger_conditions": ["no_followup_scheduled", "any_high_risk", "any_moderate_risk"],
        "estimated_duration_minutes": 15,
        "template_message": (
            "FOLLOW-UP SCHEDULING: Please schedule a PCP or specialist follow-up for patient {patient_id} "
            "within 7 days of discharge ({discharge_date}). Preferred contact: {preferred_language} speaker."
        ),
    },
    {
        "id": "INT003",
        "name": "Specialist Follow-Up Scheduling",
        "type": "care_coordination",
        "responsible_team": "Care Coordinator",
        "priority": "MEDIUM",
        "evidence_base": "AHA/ACC heart failure guidelines; specialist follow-up reduces HF readmission",
        "trigger_conditions": ["complex_diagnosis", "high_cci"],
        "estimated_duration_minutes": 20,
        "template_message": (
            "SPECIALIST FOLLOW-UP REQUIRED: Patient {patient_id} has high comorbidity burden (CCI={cci}). "
            "Please coordinate specialist follow-up within 14 days."
        ),
    },
    {
        "id": "INT004",
        "name": "Social Work Referral",
        "type": "social_work",
        "responsible_team": "Social Worker",
        "priority": "HIGH",
        "evidence_base": "SDOH framework; housing/food interventions reduce 30-day readmission by 15-25%",
        "trigger_conditions": ["housing_instability", "low_social_support", "uninsured"],
        "estimated_duration_minutes": 60,
        "template_message": (
            "SOCIAL WORK REFERRAL: Patient {patient_id} has identified social risk factors including "
            "housing instability. Please conduct SDOH assessment and connect with community resources."
        ),
    },
    {
        "id": "INT005",
        "name": "Home Health Referral",
        "type": "clinical",
        "responsible_team": "Discharge Planner",
        "priority": "HIGH",
        "evidence_base": "CMS Conditions of Participation; home health reduces 30-day readmission in HF/COPD",
        "trigger_conditions": ["high_cci", "complex_diagnosis", "prolonged_los"],
        "estimated_duration_minutes": 45,
        "template_message": (
            "HOME HEALTH REFERRAL: Patient {patient_id} meets criteria for home health services. "
            "Please initiate home health referral for skilled nursing/therapy assessment."
        ),
    },
    {
        "id": "INT006",
        "name": "Remote Patient Monitoring Enrollment",
        "type": "technology",
        "responsible_team": "Telehealth Team",
        "priority": "MEDIUM",
        "evidence_base": "RPM shown to reduce HF and COPD readmissions by 15-30% (AHA 2022)",
        "trigger_conditions": ["high_risk_diagnosis", "high_utilizer"],
        "estimated_duration_minutes": 30,
        "template_message": (
            "RPM ENROLLMENT: Patient {patient_id} is eligible for remote patient monitoring. "
            "Please initiate enrollment for 30-day post-discharge monitoring."
        ),
    },
    {
        "id": "INT007",
        "name": "Transportation Assistance Referral",
        "type": "social_work",
        "responsible_team": "Social Worker",
        "priority": "MEDIUM",
        "evidence_base": "Transportation barriers linked to 3.6M missed appointments annually (APTA 2019)",
        "trigger_conditions": ["no_transportation"],
        "estimated_duration_minutes": 20,
        "template_message": (
            "TRANSPORTATION ASSISTANCE: Patient {patient_id} lacks reliable transportation. "
            "Please arrange NEMT or transportation voucher for follow-up appointments."
        ),
    },
    {
        "id": "INT008",
        "name": "Patient Education - Warning Signs",
        "type": "education",
        "responsible_team": "Bedside Nurse / Educator",
        "priority": "MEDIUM",
        "evidence_base": "Health literacy interventions reduce ED visits by 18% (Agency for Healthcare Research)",
        "trigger_conditions": ["any_discharge"],
        "estimated_duration_minutes": 20,
        "template_message": (
            "PATIENT EDUCATION: Please provide teach-back education to patient {patient_id} on "
            "warning signs requiring ED/urgent care, medication management, and care plan adherence."
        ),
    },
    {
        "id": "INT009",
        "name": "Pharmacy Consultation for Polypharmacy",
        "type": "clinical",
        "responsible_team": "Clinical Pharmacist",
        "priority": "MEDIUM",
        "evidence_base": "Comprehensive medication review reduces drug-related readmissions by 25%",
        "trigger_conditions": ["polypharmacy_high"],
        "estimated_duration_minutes": 45,
        "template_message": (
            "PHARMACY CONSULTATION: Patient {patient_id} has {n_meds} active medications. "
            "Please schedule pharmacist consultation for comprehensive medication review and simplification."
        ),
    },
    {
        "id": "INT010",
        "name": "Community Health Worker Outreach",
        "type": "community",
        "responsible_team": "Community Health Worker",
        "priority": "MEDIUM",
        "evidence_base": "CHW outreach reduces 30-day readmission by 15-20% in vulnerable populations",
        "trigger_conditions": ["medicaid", "uninsured", "low_social_support"],
        "estimated_duration_minutes": 60,
        "template_message": (
            "CHW OUTREACH: Patient {patient_id} would benefit from community health worker support. "
            "Please assign a CHW for post-discharge outreach within 48 hours."
        ),
    },
]


def get_intervention_by_id(intervention_id: str) -> Dict:
    """Look up an intervention by ID."""
    for intervention in INTERVENTION_LIBRARY:
        if intervention["id"] == intervention_id:
            return intervention
    return {}


def get_interventions_by_trigger(trigger: str) -> List[Dict]:
    """Get all interventions that apply to a specific trigger condition."""
    return [i for i in INTERVENTION_LIBRARY if trigger in i["trigger_conditions"]]


def format_intervention_message(intervention: Dict, patient_data: Dict) -> str:
    """Format intervention template message with patient data."""
    try:
        return intervention["template_message"].format(
            patient_id=patient_data.get("patient_id", "UNKNOWN"),
            discharge_date=patient_data.get("discharge_date", "N/A"),
            preferred_language=patient_data.get("preferred_language", "English"),
            cci=patient_data.get("charlson_comorbidity_index", "N/A"),
            n_meds=patient_data.get("num_active_medications", "N/A"),
        )
    except Exception:
        return intervention.get("template_message", "")
