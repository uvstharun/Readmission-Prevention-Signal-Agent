"""
FastAPI route definitions.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from src.agents.orchestrator import ReadmissionPreventionOrchestrator
from src.monitoring.post_discharge_monitor import PostDischargeMonitor
from src.utils.database import (
    get_patient_risk_history, get_patient_interventions,
    get_active_watchlist, get_dashboard_summary,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Lazy-load orchestrator to avoid startup failures if model not trained
_orchestrator = None
_monitor = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ReadmissionPreventionOrchestrator()
    return _orchestrator


def get_monitor():
    global _monitor
    if _monitor is None:
        _monitor = PostDischargeMonitor()
    return _monitor


class DischargeEvent(BaseModel):
    patient_id: str
    age: int
    gender: str
    race: str
    ethnicity: str = "Non-Hispanic"
    preferred_language: str = "English"
    zip_code: str = "10001"
    admission_date: str
    discharge_date: str
    length_of_stay_days: int
    admission_type: str = "Emergency"
    discharge_disposition: str = "Home"
    attending_department: str = "Medicine"
    primary_diagnosis_code: str
    secondary_diagnosis_codes: str = ""
    primary_procedure_code: str = ""
    charlson_comorbidity_index: int = 0
    prior_admissions_6mo: int = 0
    prior_ed_visits_6mo: int = 0
    prior_readmissions_1yr: int = 0
    num_active_medications: int = 0
    high_risk_medication_flag: int = 0
    insurance_type: str = "Medicare"
    housing_stability_flag: int = 1
    transportation_access_flag: int = 1
    social_support_score: int = 5
    followup_appointment_scheduled: int = 0
    pcp_assigned_flag: int = 1
    discharge_instructions_given: int = 1
    use_llm: bool = True


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "readmission-prevention-agent"}


@router.post("/discharge")
def process_discharge(event: DischargeEvent):
    """Process a new discharge event through the full agent pipeline."""
    try:
        orch = get_orchestrator()
        patient_data = event.dict()
        use_llm = patient_data.pop("use_llm", True)
        result = orch.process_discharge(patient_data, use_llm=use_llm)
        return result
    except Exception as e:
        logger.error(f"Discharge processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_id}/risk")
def get_patient_risk(patient_id: str):
    """Get risk score history for a patient."""
    history = get_patient_risk_history(patient_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"No risk data found for patient {patient_id}")
    return {"patient_id": patient_id, "risk_history": history}


@router.get("/patient/{patient_id}/interventions")
def get_patient_interventions_route(patient_id: str):
    """Get triggered interventions for a patient."""
    interventions = get_patient_interventions(patient_id)
    return {"patient_id": patient_id, "interventions": interventions}


@router.get("/watchlist")
def get_watchlist():
    """Get all active 30-day monitoring patients."""
    return get_monitor().get_watchlist_summary()


@router.get("/dashboard/summary")
def get_dashboard():
    """Get aggregate dashboard metrics."""
    return get_dashboard_summary()


@router.post("/monitoring/run-cycle")
def run_monitoring_cycle():
    """Trigger a manual monitoring cycle."""
    result = get_monitor().run_cycle()
    return result
