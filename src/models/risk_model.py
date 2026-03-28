"""
Readmission risk prediction model wrapper.
Handles model loading, prediction, and SHAP explanation generation.
"""
import pickle
import numpy as np
import pandas as pd
import shap
import os
from typing import Dict, Tuple, List
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReadmissionRiskModel:
    """
    Wrapper for the trained readmission risk model.
    Provides risk scoring and SHAP-based explainability.
    """

    def __init__(
        self,
        model_path: str = None,
        explainer_path: str = None,
    ):
        self.model_path = model_path or config.MODEL_PATH
        self.explainer_path = explainer_path or config.SHAP_EXPLAINER_PATH
        self.model = None
        self.explainer = None
        self.feature_names = None
        self._load()

    def _load(self):
        """Load model and SHAP explainer from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Train the model first.")

        with open(self.model_path, "rb") as f:
            artifact = pickle.load(f)

        if isinstance(artifact, dict):
            self.model = artifact["model"]
            self.feature_names = artifact.get("feature_names", [])
            self.threshold_high = artifact.get("threshold_high", config.RISK_THRESHOLD_HIGH)
            self.threshold_moderate = artifact.get("threshold_moderate", config.RISK_THRESHOLD_MODERATE)
        else:
            self.model = artifact
            self.feature_names = []

        logger.info(f"Loaded model from {self.model_path}")

        if os.path.exists(self.explainer_path):
            with open(self.explainer_path, "rb") as f:
                self.explainer = pickle.load(f)
            logger.info(f"Loaded SHAP explainer from {self.explainer_path}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability scores for readmission."""
        return self.model.predict_proba(X)[:, 1]

    def predict_risk_tier(self, score: float) -> str:
        """Map probability score to risk tier."""
        high = getattr(self, "threshold_high", config.RISK_THRESHOLD_HIGH)
        moderate = getattr(self, "threshold_moderate", config.RISK_THRESHOLD_MODERATE)
        if score >= high:
            return "HIGH"
        elif score >= moderate:
            return "MODERATE"
        return "LOW"

    def get_shap_values(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Generate SHAP values for feature importance."""
        if self.explainer is None:
            logger.warning("SHAP explainer not loaded")
            return np.zeros((len(X), len(self.feature_names))), self.feature_names

        shap_values = self.explainer.shap_values(X)
        # For binary classifiers, shap_values may be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values, list(X.columns)

    def explain_patient(self, X: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """
        Get top N SHAP feature contributions for a single patient.
        Returns list of dicts with feature name, value, and SHAP contribution.
        """
        shap_values, feature_names = self.get_shap_values(X)

        if len(X) == 1:
            shap_row = shap_values[0]
            feature_values = X.iloc[0].values

            contributions = []
            for i, (fname, fval, sval) in enumerate(zip(feature_names, feature_values, shap_row)):
                contributions.append({
                    "feature": fname,
                    "value": float(fval),
                    "shap_contribution": float(sval),
                    "direction": "increases" if sval > 0 else "decreases",
                })

            contributions.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)
            return contributions[:top_n]

        return []

    def score_patient(self, X: pd.DataFrame) -> Dict:
        """
        Full scoring pipeline for a single patient.
        Returns risk score, tier, and top SHAP drivers.
        """
        score = float(self.predict_proba(X)[0])
        tier = self.predict_risk_tier(score)
        top_features = self.explain_patient(X, top_n=5)

        return {
            "risk_score": round(score, 4),
            "risk_tier": tier,
            "top_risk_drivers": top_features,
        }
