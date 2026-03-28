"""
Model drift detection: monitors for distribution shift in incoming patient data.
"""
import numpy as np
from typing import Dict, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) for drift detection.
    PSI < 0.1: no drift | 0.1-0.25: slight drift | > 0.25: significant drift
    """
    expected_hist, bin_edges = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = (expected_hist + 1e-6) / len(expected)
    actual_pct = (actual_hist + 1e-6) / len(actual)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def check_score_drift(baseline_scores: List[float], recent_scores: List[float]) -> Dict:
    """Check for drift in risk score distribution."""
    if len(baseline_scores) < 10 or len(recent_scores) < 10:
        return {"status": "INSUFFICIENT_DATA", "psi": None}

    psi = compute_psi(np.array(baseline_scores), np.array(recent_scores))

    if psi < 0.1:
        status = "STABLE"
    elif psi < 0.25:
        status = "SLIGHT_DRIFT"
    else:
        status = "SIGNIFICANT_DRIFT"

    logger.info(f"Score drift check: PSI={psi:.4f} status={status}")
    return {
        "status": status,
        "psi": round(psi, 4),
        "baseline_mean": float(np.mean(baseline_scores)),
        "recent_mean": float(np.mean(recent_scores)),
        "recommendation": "Retrain model" if status == "SIGNIFICANT_DRIFT" else "Continue monitoring",
    }
