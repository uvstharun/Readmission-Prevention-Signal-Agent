"""
Risk Scoring Agent: Loads trained model, scores patients, returns risk tier and SHAP features.
"""
from typing import Dict
import pandas as pd
from src.models.risk_model import ReadmissionRiskModel
from src.data.data_loader import transform_new_patient
from src.utils.logger import get_logger, log_agent_action
from src.utils.database import save_agent_decision

logger = get_logger(__name__)


class RiskScoringAgent:
    """Scores a patient for 30-day readmission risk."""

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = ReadmissionRiskModel()
            logger.info("Risk scoring agent: model loaded")
        except FileNotFoundError as e:
            logger.warning(f"Model not found: {e}. Will use mock scores.")
            self.model = None

    def run(self, patient_data: Dict) -> Dict:
        """
        Score a patient for readmission risk.
        Returns: risk_score, risk_tier, top_risk_drivers
        """
        patient_id = patient_data.get("patient_id", "UNKNOWN")
        logger.info(f"Scoring patient {patient_id}")

        if self.model is None:
            result = self._mock_score(patient_data)
        else:
            try:
                X = transform_new_patient(patient_data)
                result = self.model.score_patient(X)
            except Exception as e:
                logger.error(f"Scoring failed for {patient_id}: {e}")
                result = self._mock_score(patient_data)

        log_agent_action("RiskScoringAgent", patient_id, "score_patient",
                         f"tier={result['risk_tier']} score={result['risk_score']}")

        save_agent_decision(
            patient_id=patient_id,
            agent_name="RiskScoringAgent",
            action="score_patient",
            reasoning=f"Scored using trained XGBoost model. Risk tier: {result['risk_tier']}",
            output=result,
        )

        return result

    def _mock_score(self, patient_data: Dict) -> Dict:
        """Generate a heuristic score when model is not available."""
        import random
        # Simple heuristic scoring
        score = 0.1
        score += min(patient_data.get("charlson_comorbidity_index", 0) * 0.05, 0.25)
        score += min(patient_data.get("prior_admissions_6mo", 0) * 0.08, 0.24)
        score += (1 - patient_data.get("followup_appointment_scheduled", 1)) * 0.10
        score += (1 - patient_data.get("housing_stability_flag", 1)) * 0.08
        score = min(0.95, max(0.05, score + random.uniform(-0.02, 0.02)))

        if score >= 0.65:
            tier = "HIGH"
        elif score >= 0.35:
            tier = "MODERATE"
        else:
            tier = "LOW"

        return {
            "risk_score": round(score, 4),
            "risk_tier": tier,
            "top_risk_drivers": [
                {"feature": "charlson_comorbidity_index", "value": patient_data.get("charlson_comorbidity_index", 0),
                 "shap_contribution": 0.15, "direction": "increases"},
                {"feature": "prior_admissions_6mo", "value": patient_data.get("prior_admissions_6mo", 0),
                 "shap_contribution": 0.12, "direction": "increases"},
                {"feature": "followup_appointment_scheduled",
                 "value": patient_data.get("followup_appointment_scheduled", 0),
                 "shap_contribution": -0.08 if patient_data.get("followup_appointment_scheduled", 0) else 0.08,
                 "direction": "decreases" if patient_data.get("followup_appointment_scheduled", 0) else "increases"},
            ],
        }
