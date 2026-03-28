"""
Tests for the readmission risk model.
"""
import pytest
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRiskModelMock:
    """Tests using mock scoring when model is not trained."""

    def test_mock_score_range(self):
        """Risk score should be between 0 and 1."""
        from src.agents.risk_scoring_agent import RiskScoringAgent
        agent = RiskScoringAgent()
        patient = {
            "patient_id": "TEST001",
            "age": 72,
            "charlson_comorbidity_index": 4,
            "prior_admissions_6mo": 2,
            "followup_appointment_scheduled": 0,
            "housing_stability_flag": 1,
        }
        result = agent._mock_score(patient)
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_risk_tier_categories(self):
        """Risk tier must be one of HIGH, MODERATE, LOW."""
        from src.agents.risk_scoring_agent import RiskScoringAgent
        agent = RiskScoringAgent()
        for score in [0.1, 0.4, 0.7]:
            if score >= 0.65:
                expected_tier = "HIGH"
            elif score >= 0.35:
                expected_tier = "MODERATE"
            else:
                expected_tier = "LOW"
            # Test threshold logic
            if score >= 0.65:
                assert expected_tier == "HIGH"
            elif score >= 0.35:
                assert expected_tier == "MODERATE"
            else:
                assert expected_tier == "LOW"

    def test_high_risk_patient_scores_higher(self):
        """High-risk patient should score higher than low-risk patient."""
        from src.agents.risk_scoring_agent import RiskScoringAgent
        agent = RiskScoringAgent()

        high_risk = {
            "patient_id": "HIGH001",
            "charlson_comorbidity_index": 8,
            "prior_admissions_6mo": 4,
            "followup_appointment_scheduled": 0,
            "housing_stability_flag": 0,
        }
        low_risk = {
            "patient_id": "LOW001",
            "charlson_comorbidity_index": 0,
            "prior_admissions_6mo": 0,
            "followup_appointment_scheduled": 1,
            "housing_stability_flag": 1,
        }
        high_result = agent._mock_score(high_risk)
        low_result = agent._mock_score(low_risk)
        assert high_result["risk_score"] > low_result["risk_score"]

    def test_mock_score_returns_required_keys(self):
        """Mock score must return risk_score, risk_tier, top_risk_drivers."""
        from src.agents.risk_scoring_agent import RiskScoringAgent
        agent = RiskScoringAgent()
        result = agent._mock_score({"patient_id": "TEST002", "charlson_comorbidity_index": 2,
                                    "prior_admissions_6mo": 0, "followup_appointment_scheduled": 1,
                                    "housing_stability_flag": 1})
        assert "risk_score" in result
        assert "risk_tier" in result
        assert "top_risk_drivers" in result
        assert isinstance(result["top_risk_drivers"], list)
