"""
Feedback loop: captures actual readmission outcomes and feeds back into model monitoring.
"""
from typing import Dict, List
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def record_actual_outcome(patient_id: str, readmitted: bool, readmission_date: str = None,
                           readmission_diagnosis: str = None):
    """Record actual 30-day readmission outcome for a patient."""
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO readmission_outcomes
            (patient_id, actual_readmission, readmission_date, readmission_diagnosis)
            VALUES (?, ?, ?, ?)
        """, (patient_id, int(readmitted), readmission_date, readmission_diagnosis))
    logger.info(f"Recorded outcome for {patient_id}: readmitted={readmitted}")


def compute_model_performance() -> Dict:
    """Compute actual vs predicted performance from outcome data."""
    with get_connection() as conn:
        outcomes = conn.execute("""
            SELECT ro.patient_id, ro.actual_readmission, rs.risk_score, rs.risk_tier
            FROM readmission_outcomes ro
            JOIN risk_scores rs ON ro.patient_id = rs.patient_id
        """).fetchall()

    if not outcomes:
        return {"message": "No outcome data available yet"}

    outcomes = [dict(o) for o in outcomes]
    total = len(outcomes)
    actual_readmissions = sum(o["actual_readmission"] for o in outcomes)

    tier_stats = {}
    for tier in ["HIGH", "MODERATE", "LOW"]:
        tier_outcomes = [o for o in outcomes if o["risk_tier"] == tier]
        if tier_outcomes:
            tier_stats[tier] = {
                "count": len(tier_outcomes),
                "actual_readmission_rate": sum(o["actual_readmission"] for o in tier_outcomes) / len(tier_outcomes),
            }

    return {
        "total_outcomes": total,
        "overall_readmission_rate": actual_readmissions / total if total > 0 else 0,
        "by_risk_tier": tier_stats,
    }
