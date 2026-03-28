"""
Central configuration management using environment variables.
"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class AppConfig:
    # API Keys
    ANTHROPIC_API_KEY: str = ""

    # Database
    DATABASE_URL: str = "sqlite:///readmission_agent.db"

    # Model paths
    MODEL_PATH: str = "models/readmission_risk_model.pkl"
    SHAP_EXPLAINER_PATH: str = "models/shap_explainer.pkl"
    ENCODERS_PATH: str = "models/encoders.pkl"
    SCALER_PATH: str = "models/scaler.pkl"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/agent.log"

    # Risk thresholds
    RISK_THRESHOLD_HIGH: float = 0.65
    RISK_THRESHOLD_MODERATE: float = 0.35

    # Monitoring
    MONITORING_WINDOW_DAYS: int = 30

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    STREAMLIT_PORT: int = 8501


def _load_config() -> AppConfig:
    return AppConfig(
        ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""),
        DATABASE_URL=os.getenv("DATABASE_URL", "sqlite:///readmission_agent.db"),
        MODEL_PATH=os.getenv("MODEL_PATH", "models/readmission_risk_model.pkl"),
        SHAP_EXPLAINER_PATH=os.getenv("SHAP_EXPLAINER_PATH", "models/shap_explainer.pkl"),
        ENCODERS_PATH=os.getenv("ENCODERS_PATH", "models/encoders.pkl"),
        SCALER_PATH=os.getenv("SCALER_PATH", "models/scaler.pkl"),
        LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
        LOG_FILE=os.getenv("LOG_FILE", "logs/agent.log"),
        RISK_THRESHOLD_HIGH=float(os.getenv("RISK_THRESHOLD_HIGH", "0.65")),
        RISK_THRESHOLD_MODERATE=float(os.getenv("RISK_THRESHOLD_MODERATE", "0.35")),
        MONITORING_WINDOW_DAYS=int(os.getenv("MONITORING_WINDOW_DAYS", "30")),
        API_HOST=os.getenv("API_HOST", "0.0.0.0"),
        API_PORT=int(os.getenv("API_PORT", "8000")),
        STREAMLIT_PORT=int(os.getenv("STREAMLIT_PORT", "8501")),
    )


config = _load_config()
