"""
Tests for individual agents.
"""
import pytest
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_PATIENT = {
    "patient_id": "TEST100",
    "age": 68,
    "gender": "Female",
    "race": "White",
    "ethnicity": "Non-Hispanic",
    "preferred_language": "English",
    "zip_code": "10001",
    "admission_date": "2024-01-15",
    "discharge_date": "2024-01-20",
    "length_of_stay_days": 5,
    "admission_type": "Emergency",
    "discharge_disposition": "Home",
    "attending_department": "Cardiology",
    "primary_diagnosis_code": "I50.20",
    "secondary_diagnosis_codes": "E11.9|N18.3",
    "primary_procedure_code": "93306",
    "charlson_comorbidity_index": 4,
    "prior_admissions_6mo": 2,
    "prior_ed_visits_6mo": 1,
    "prior_readmissions_1yr": 1,
    "num_active_medications": 9,
    "high_risk_medication_flag": 1,
    "insurance_type": "Medicare",
    "housing_stability_flag": 1,
    "transportation_access_flag": 0,
    "social_support_score": 3,
    "followup_appointment_scheduled": 0,
    "pcp_assigned_flag": 1,
    "discharge_instructions_given": 1,
}


class TestPatientContextAgent:
    def test_returns_required_keys(self):
        from src.agents.patient_context_agent import PatientContextAgent
        agent = PatientContextAgent()
        result = agent.run(SAMPLE_PATIENT)
        assert "demographics" in result
        assert "clinical_summary" in result
        assert "utilization_summary" in result
        assert "risk_flags" in result

    def test_identifies_risk_flags(self):
        from src.agents.patient_context_agent import PatientContextAgent
        agent = PatientContextAgent()
        result = agent.run(SAMPLE_PATIENT)
        flags = result["risk_flags"]
        assert "HIGH_COMORBIDITY_BURDEN" in flags
        assert "HIGH_RISK_MEDICATIONS" in flags
        assert "NO_FOLLOWUP_SCHEDULED" in flags

    def test_handles_missing_fields(self):
        from src.agents.patient_context_agent import PatientContextAgent
        agent = PatientContextAgent()
        minimal = {"patient_id": "MIN001"}
        result = agent.run(minimal)
        assert "demographics" in result


class TestCareGapAgent:
    def test_identifies_care_gaps(self):
        from src.agents.care_gap_agent import CareGapAgent
        agent = CareGapAgent()
        risk_result = {"risk_tier": "HIGH", "risk_score": 0.72}
        result = agent.run(SAMPLE_PATIENT, risk_result)
        assert "care_gaps" in result
        assert result["total_gaps"] > 0

    def test_no_followup_flagged(self):
        from src.agents.care_gap_agent import CareGapAgent
        agent = CareGapAgent()
        risk_result = {"risk_tier": "HIGH", "risk_score": 0.72}
        result = agent.run(SAMPLE_PATIENT, risk_result)
        gap_names = [g["gap"] for g in result["care_gaps"]]
        assert any("Follow-Up" in g for g in gap_names)

    def test_all_gaps_have_required_fields(self):
        from src.agents.care_gap_agent import CareGapAgent
        agent = CareGapAgent()
        risk_result = {"risk_tier": "MODERATE", "risk_score": 0.45}
        result = agent.run(SAMPLE_PATIENT, risk_result)
        for gap in result["care_gaps"]:
            assert "gap" in gap
            assert "priority" in gap
            assert "rationale" in gap
            assert gap["priority"] in ["HIGH", "MEDIUM", "LOW"]


class TestRiskScoringAgent:
    def test_scores_patient(self):
        from src.agents.risk_scoring_agent import RiskScoringAgent
        agent = RiskScoringAgent()
        result = agent.run(SAMPLE_PATIENT)
        assert "risk_score" in result
        assert "risk_tier" in result
        assert 0 <= result["risk_score"] <= 1

    def test_tier_is_valid(self):
        from src.agents.risk_scoring_agent import RiskScoringAgent
        agent = RiskScoringAgent()
        result = agent.run(SAMPLE_PATIENT)
        assert result["risk_tier"] in ["HIGH", "MODERATE", "LOW"]
