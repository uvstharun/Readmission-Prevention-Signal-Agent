"""
Alert engine: formats and delivers structured case manager alerts.
"""
from typing import Dict, List
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)

RISK_TIER_COLORS = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}


def format_case_manager_alert(
    patient_data: Dict,
    risk_result: Dict,
    clinical_narrative: Dict,
    care_gaps: List[Dict],
    interventions: List[Dict],
) -> Dict:
    """
    Format a complete structured alert for the case manager.
    Includes patient summary, risk score, clinical narrative, care gaps, and action checklist.
    """
    patient_id = patient_data.get("patient_id", "UNKNOWN")
    risk_tier = risk_result.get("risk_tier", "LOW")
    risk_score = risk_result.get("risk_score", 0)
    tier_icon = RISK_TIER_COLORS.get(risk_tier, "⚪")

    alert = {
        "alert_id": f"ALERT-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "generated_at": datetime.now().isoformat(),
        "alert_type": f"{risk_tier}_RISK_DISCHARGE",
        "priority": risk_tier,
        "patient_summary": {
            "patient_id": patient_id,
            "age": patient_data.get("age"),
            "gender": patient_data.get("gender"),
            "primary_diagnosis": patient_data.get("primary_diagnosis_code"),
            "discharge_date": patient_data.get("discharge_date"),
            "discharge_disposition": patient_data.get("discharge_disposition"),
            "attending_department": patient_data.get("attending_department"),
            "insurance": patient_data.get("insurance_type"),
        },
        "risk_assessment": {
            "risk_score": risk_score,
            "risk_score_pct": f"{risk_score:.1%}",
            "risk_tier": risk_tier,
            "tier_display": f"{tier_icon} {risk_tier} RISK",
            "charlson_comorbidity_index": patient_data.get("charlson_comorbidity_index"),
            "prior_admissions_6mo": patient_data.get("prior_admissions_6mo"),
        },
        "clinical_narrative": clinical_narrative,
        "care_gaps": {
            "total_gaps": len(care_gaps),
            "high_priority_gaps": sum(1 for g in care_gaps if g.get("priority") == "HIGH"),
            "gaps": care_gaps,
        },
        "action_checklist": [
            {
                "action_number": i + 1,
                "intervention": interv.get("name"),
                "responsible_team": interv.get("responsible_team"),
                "priority": interv.get("priority"),
                "type": interv.get("type"),
                "status": "PENDING",
            }
            for i, interv in enumerate(interventions)
        ],
        "monitoring": {
            "watchlist": risk_tier in ["HIGH", "MODERATE"],
            "monitoring_days": 30 if risk_tier in ["HIGH", "MODERATE"] else None,
            "monitoring_frequency": "Every 3 days" if risk_tier == "HIGH" else (
                "Every 7 days" if risk_tier == "MODERATE" else "Not enrolled"
            ),
        },
        "header_text": _build_header(patient_id, risk_tier, risk_score, patient_data),
    }

    logger.info(f"Formatted case manager alert for {patient_id}: {risk_tier} risk, "
                f"{len(care_gaps)} gaps, {len(interventions)} interventions")

    return alert


def _build_header(patient_id: str, risk_tier: str, risk_score: float, patient_data: Dict) -> str:
    """Build human-readable alert header."""
    icon = RISK_TIER_COLORS.get(risk_tier, "⚪")
    return (
        f"{icon} {risk_tier} READMISSION RISK ALERT | {patient_id} | "
        f"Score: {risk_score:.1%} | "
        f"Dx: {patient_data.get('primary_diagnosis_code', 'N/A')} | "
        f"Discharged: {patient_data.get('discharge_date', 'N/A')}"
    )
