"""
Risk stratification routing: maps risk tiers to evidence-based intervention bundles.
"""
from typing import Dict, List
from src.workflows.intervention_library import INTERVENTION_LIBRARY, format_intervention_message
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Risk tier intervention bundles
INTERVENTION_BUNDLES = {
    "HIGH": {
        "description": "Intensive care transition bundle for high-risk patients",
        "mandatory_interventions": ["INT001", "INT002", "INT004", "INT008"],
        "conditional_interventions": ["INT003", "INT005", "INT006", "INT009", "INT010"],
        "monitoring_frequency": "every_3_days",
        "case_manager_required": True,
        "physician_notification_required": True,
    },
    "MODERATE": {
        "description": "Standard care transition bundle for moderate-risk patients",
        "mandatory_interventions": ["INT002", "INT008"],
        "conditional_interventions": ["INT001", "INT004", "INT007", "INT010"],
        "monitoring_frequency": "every_7_days",
        "case_manager_required": True,
        "physician_notification_required": False,
    },
    "LOW": {
        "description": "Basic care transition checklist for low-risk patients",
        "mandatory_interventions": ["INT008"],
        "conditional_interventions": ["INT002"],
        "monitoring_frequency": "at_30_days",
        "case_manager_required": False,
        "physician_notification_required": False,
    },
}

# Map care gap IDs to intervention IDs
GAP_TO_INTERVENTION = {
    "CG001": "INT002",  # No follow-up → Schedule follow-up
    "CG002": "INT002",  # No PCP → Coordinate care
    "CG003": "INT008",  # No instructions → Education
    "CG004": "INT001",  # Polypharmacy → Medication reconciliation
    "CG005": "INT001",  # High-risk meds → Med review
    "CG006": "INT004",  # Housing → Social work
    "CG007": "INT007",  # Transport → Transportation assistance
    "CG008": "INT010",  # Low support → CHW
    "CG009": "INT005",  # High complexity → Home health
    "CG010": "INT004",  # Financial → Social work
}


def _lookup_intervention(intervention_id: str) -> Dict:
    """Lookup intervention definition by ID."""
    for i in INTERVENTION_LIBRARY:
        if i["id"] == intervention_id:
            return i.copy()
    return {}


def get_intervention_bundle(risk_tier: str, care_gaps: List[Dict]) -> List[Dict]:
    """
    Build the complete intervention list for a patient based on risk tier and care gaps.
    Deduplicates interventions.
    """
    bundle_config = INTERVENTION_BUNDLES.get(risk_tier, INTERVENTION_BUNDLES["LOW"])
    seen_ids = set()
    interventions = []

    # Add mandatory interventions
    for int_id in bundle_config["mandatory_interventions"]:
        if int_id not in seen_ids:
            intervention = _lookup_intervention(int_id)
            if intervention:
                intervention["triggered_by"] = "risk_tier_bundle"
                interventions.append(intervention)
                seen_ids.add(int_id)

    # Add interventions from care gaps
    for gap in care_gaps:
        gap_id = gap.get("gap_id", "")
        int_id = GAP_TO_INTERVENTION.get(gap_id)
        if int_id and int_id not in seen_ids:
            intervention = _lookup_intervention(int_id)
            if intervention:
                intervention["triggered_by"] = f"care_gap:{gap_id}"
                interventions.append(intervention)
                seen_ids.add(int_id)

    logger.info(f"Built intervention bundle for {risk_tier} risk: {len(interventions)} interventions")
    return interventions


def get_bundle_summary(risk_tier: str) -> Dict:
    """Get the intervention bundle configuration summary for a risk tier."""
    return INTERVENTION_BUNDLES.get(risk_tier, {})
