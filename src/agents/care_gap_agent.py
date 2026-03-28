"""
Care Gap Agent: Identifies missing evidence-based care transition interventions.
"""
from typing import Dict, List
from src.utils.logger import get_logger, log_agent_action
from src.utils.database import save_agent_decision

logger = get_logger(__name__)


class CareGapAgent:
    """Evaluates which evidence-based care transition interventions are missing."""

    # Evidence-based care gap criteria
    CARE_GAP_CHECKS = [
        {
            "gap_id": "CG001",
            "name": "No Follow-Up Appointment Scheduled",
            "check": lambda d: not d.get("followup_appointment_scheduled", 0),
            "priority": "HIGH",
            "rationale": "Patients without scheduled follow-up within 7 days have 2x readmission risk (Project RED evidence)",
            "recommendation": "Schedule PCP or specialist follow-up within 7 days of discharge",
        },
        {
            "gap_id": "CG002",
            "name": "No PCP Assigned",
            "check": lambda d: not d.get("pcp_assigned_flag", 0),
            "priority": "HIGH",
            "rationale": "Lack of primary care attribution is associated with increased fragmentation and readmission",
            "recommendation": "Assign or confirm PCP and provide warm handoff with discharge summary",
        },
        {
            "gap_id": "CG003",
            "name": "Discharge Instructions Not Provided",
            "check": lambda d: not d.get("discharge_instructions_given", 0),
            "priority": "MEDIUM",
            "rationale": "Inadequate health literacy support increases medication errors and misunderstanding of warning signs",
            "recommendation": "Provide and review teach-back discharge instructions covering medications, warning signs, and follow-up plan",
        },
        {
            "gap_id": "CG004",
            "name": "Polypharmacy Without Reconciliation Review",
            "check": lambda d: d.get("num_active_medications", 0) >= 5,
            "priority": "HIGH",
            "rationale": "5+ medications at discharge is a marker for adverse drug events and 30-day readmission (Beers Criteria)",
            "recommendation": "Complete medication reconciliation review with pharmacist before discharge",
        },
        {
            "gap_id": "CG005",
            "name": "High-Risk Medication Without Counseling",
            "check": lambda d: bool(d.get("high_risk_medication_flag", 0)),
            "priority": "HIGH",
            "rationale": "Anticoagulants, insulin, and opioids are leading causes of adverse drug events post-discharge",
            "recommendation": "Provide dedicated high-risk medication counseling and ensure monitoring plan is in place",
        },
        {
            "gap_id": "CG006",
            "name": "Housing Instability Without Social Work Referral",
            "check": lambda d: not d.get("housing_stability_flag", 1),
            "priority": "HIGH",
            "rationale": "Housing instability is a primary SDOH driver of readmission and worse health outcomes",
            "recommendation": "Refer to social work for housing assessment and community resources",
        },
        {
            "gap_id": "CG007",
            "name": "Transportation Barrier Not Addressed",
            "check": lambda d: not d.get("transportation_access_flag", 1),
            "priority": "MEDIUM",
            "rationale": "Transportation barriers result in missed follow-up appointments and medication non-adherence",
            "recommendation": "Arrange transportation assistance for follow-up appointments",
        },
        {
            "gap_id": "CG008",
            "name": "Low Social Support Without Community Health Worker Referral",
            "check": lambda d: d.get("social_support_score", 5) <= 3,
            "priority": "MEDIUM",
            "rationale": "Social isolation is associated with worse adherence and higher readmission risk",
            "recommendation": "Refer to Community Health Worker or care navigator for ongoing support",
        },
        {
            "gap_id": "CG009",
            "name": "High Complexity Patient Without Home Health Referral",
            "check": lambda d: (d.get("charlson_comorbidity_index", 0) >= 4 and
                                d.get("discharge_disposition") in ["Home", "AMA"]),
            "priority": "MEDIUM",
            "rationale": "High-complexity patients discharged home benefit from skilled nursing or home health follow-up",
            "recommendation": "Evaluate for home health referral for wound care, IV therapy, or complex medication management",
        },
        {
            "gap_id": "CG010",
            "name": "Uninsured/Medicaid Patient Without Financial Counseling",
            "check": lambda d: d.get("insurance_type") in ["Uninsured", "Medicaid"],
            "priority": "MEDIUM",
            "rationale": "Financial barriers to medication and follow-up care drive preventable readmissions",
            "recommendation": "Connect with financial counselor for assistance programs and medication access",
        },
    ]

    def run(self, patient_data: Dict, risk_result: Dict) -> Dict:
        """
        Identify care gaps for a patient.
        Returns prioritized list of gaps with clinical rationale.
        """
        patient_id = patient_data.get("patient_id", "UNKNOWN")
        risk_tier = risk_result.get("risk_tier", "LOW")

        identified_gaps = []
        for check in self.CARE_GAP_CHECKS:
            try:
                if check["check"](patient_data):
                    identified_gaps.append({
                        "gap_id": check["gap_id"],
                        "gap": check["name"],
                        "priority": check["priority"],
                        "rationale": check["rationale"],
                        "recommended_intervention": check["recommendation"],
                    })
            except Exception:
                pass

        # Sort by priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        identified_gaps.sort(key=lambda x: priority_order.get(x["priority"], 3))

        # Overall severity
        high_count = sum(1 for g in identified_gaps if g["priority"] == "HIGH")
        if high_count >= 3:
            severity = "HIGH"
        elif high_count >= 1:
            severity = "MODERATE"
        else:
            severity = "LOW"

        result = {
            "patient_id": patient_id,
            "care_gaps": identified_gaps,
            "total_gaps": len(identified_gaps),
            "high_priority_gaps": high_count,
            "care_gap_severity": severity,
        }

        log_agent_action("CareGapAgent", patient_id, "identify_gaps",
                         f"gaps={len(identified_gaps)} severity={severity}")

        save_agent_decision(
            patient_id=patient_id,
            agent_name="CareGapAgent",
            action="identify_gaps",
            reasoning=f"Evaluated {len(self.CARE_GAP_CHECKS)} evidence-based criteria. Found {len(identified_gaps)} gaps.",
            output=result,
        )

        return result
