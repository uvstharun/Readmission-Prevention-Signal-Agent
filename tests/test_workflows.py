"""
Tests for workflow components: intervention library, risk stratification, alert engine.
"""
import pytest
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestInterventionLibrary:
    def test_library_has_10_interventions(self):
        from src.workflows.intervention_library import INTERVENTION_LIBRARY
        assert len(INTERVENTION_LIBRARY) == 10

    def test_each_intervention_has_required_fields(self):
        from src.workflows.intervention_library import INTERVENTION_LIBRARY
        required = ["id", "name", "type", "responsible_team", "priority", "evidence_base"]
        for intervention in INTERVENTION_LIBRARY:
            for field in required:
                assert field in intervention, f"Missing {field} in {intervention.get('id')}"

    def test_get_intervention_by_id(self):
        from src.workflows.intervention_library import get_intervention_by_id
        result = get_intervention_by_id("INT001")
        assert result["name"] == "Medication Reconciliation Review"

    def test_missing_id_returns_empty(self):
        from src.workflows.intervention_library import get_intervention_by_id
        result = get_intervention_by_id("INT999")
        assert result == {}


class TestRiskStratification:
    def test_high_risk_bundle_has_interventions(self):
        from src.workflows.risk_stratification import get_intervention_bundle
        bundle = get_intervention_bundle("HIGH", [])
        assert len(bundle) > 0

    def test_high_risk_gets_more_than_low(self):
        from src.workflows.risk_stratification import get_intervention_bundle
        high_bundle = get_intervention_bundle("HIGH", [])
        low_bundle = get_intervention_bundle("LOW", [])
        assert len(high_bundle) >= len(low_bundle)

    def test_care_gap_adds_intervention(self):
        from src.workflows.risk_stratification import get_intervention_bundle
        gaps = [{"gap_id": "CG006", "gap": "Housing Instability", "priority": "HIGH"}]
        bundle_without_gap = get_intervention_bundle("MODERATE", [])
        bundle_with_gap = get_intervention_bundle("MODERATE", gaps)
        # Bundle with gap should include social work referral (INT004)
        ids_with = [i["id"] for i in bundle_with_gap]
        assert "INT004" in ids_with


class TestAlertEngine:
    def test_alert_has_required_fields(self):
        from src.workflows.alert_engine import format_case_manager_alert
        patient_data = {
            "patient_id": "TEST200",
            "age": 70,
            "gender": "Male",
            "primary_diagnosis_code": "I50.20",
            "discharge_date": "2024-01-20",
            "discharge_disposition": "Home",
            "attending_department": "Cardiology",
            "insurance_type": "Medicare",
            "charlson_comorbidity_index": 4,
            "prior_admissions_6mo": 2,
        }
        risk_result = {"risk_score": 0.75, "risk_tier": "HIGH"}
        narrative = {"risk_summary": "High risk patient"}
        alert = format_case_manager_alert(patient_data, risk_result, narrative, [], [])
        assert "alert_id" in alert
        assert "patient_summary" in alert
        assert "risk_assessment" in alert
        assert "action_checklist" in alert

    def test_high_risk_alert_priority(self):
        from src.workflows.alert_engine import format_case_manager_alert
        patient_data = {"patient_id": "TEST201", "age": 80}
        risk_result = {"risk_score": 0.80, "risk_tier": "HIGH"}
        alert = format_case_manager_alert(patient_data, risk_result, {}, [], [])
        assert alert["priority"] == "HIGH"
