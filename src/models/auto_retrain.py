"""
Auto-retraining: monitors model drift and triggers retraining when PSI > threshold.
Wires together the drift detector and model trainer.
"""
import os
import json
import pickle
import numpy as np
from datetime import datetime
from src.monitoring.drift_detector import check_score_drift
from src.models.model_trainer import run_training_pipeline
from src.data.feature_engineering import run_pipeline as run_feature_pipeline
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)

PSI_RETRAIN_THRESHOLD = 0.25
BASELINE_SCORES_PATH = "models/baseline_scores.pkl"
RETRAIN_LOG_PATH = "models/retrain_log.json"


def load_baseline_scores() -> list:
    """Load baseline risk score distribution."""
    if not os.path.exists(BASELINE_SCORES_PATH):
        return []
    with open(BASELINE_SCORES_PATH, "rb") as f:
        return pickle.load(f)


def save_baseline_scores(scores: list):
    """Save baseline risk scores after initial training."""
    os.makedirs("models", exist_ok=True)
    with open(BASELINE_SCORES_PATH, "wb") as f:
        pickle.dump(scores, f)
    logger.info(f"Saved {len(scores)} baseline scores")


def collect_recent_scores(n: int = 500) -> list:
    """Collect recent risk scores from the database."""
    try:
        from src.utils.database import _risk_scores
        recent = [r["risk_score"] for r in _risk_scores[-n:]]
        return recent
    except Exception as e:
        logger.warning(f"Could not load recent scores: {e}")
        return []


def log_retrain_event(reason: str, psi: float, new_auc: float):
    """Append a retraining event to the retrain log."""
    log = []
    if os.path.exists(RETRAIN_LOG_PATH):
        with open(RETRAIN_LOG_PATH) as f:
            log = json.load(f)

    log.append({
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "psi_score": psi,
        "new_auc": new_auc,
    })

    with open(RETRAIN_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    logger.info(f"Retrain event logged: PSI={psi:.4f} | New AUC={new_auc:.4f}")


def check_and_retrain(force: bool = False) -> dict:
    """
    Check for model drift and retrain if needed.

    Args:
        force: Skip drift check and retrain unconditionally.

    Returns:
        Dict with action taken and metrics.
    """
    logger.info("Running auto-retrain check...")

    baseline_scores = load_baseline_scores()
    recent_scores = collect_recent_scores(n=500)

    if not force:
        # Not enough data to check drift
        if len(baseline_scores) < 50 or len(recent_scores) < 50:
            logger.info(f"Insufficient data for drift check "
                        f"(baseline={len(baseline_scores)}, recent={len(recent_scores)})")
            return {"action": "skipped", "reason": "insufficient_data"}

        # Check PSI drift
        drift_result = check_score_drift(baseline_scores, recent_scores)
        psi = drift_result.get("psi", 0) or 0
        status = drift_result.get("status", "STABLE")

        logger.info(f"Drift check: PSI={psi:.4f} status={status}")

        if status != "SIGNIFICANT_DRIFT" and psi < PSI_RETRAIN_THRESHOLD:
            logger.info("No significant drift detected — skipping retrain")
            return {"action": "skipped", "reason": "no_drift", "psi": psi, "status": status}

        logger.warning(f"Significant drift detected (PSI={psi:.4f}) — triggering retrain")
    else:
        psi = 0.0
        logger.info("Force retrain requested")

    # Re-run feature engineering to pick up new data
    logger.info("Re-running feature engineering pipeline...")
    run_feature_pipeline()

    # Retrain
    logger.info("Retraining models...")
    report = run_training_pipeline()
    new_auc = report.get("best_model_metrics", {}).get("test_auc_roc", 0)

    # Update baseline scores with new distribution
    if recent_scores:
        save_baseline_scores(recent_scores)

    log_retrain_event(
        reason="force" if force else "drift_detected",
        psi=psi,
        new_auc=new_auc,
    )

    return {
        "action": "retrained",
        "psi": psi,
        "new_best_model": report.get("best_model"),
        "new_auc": new_auc,
    }


def initialize_baseline(scores: list):
    """Call this after initial training to set the baseline score distribution."""
    save_baseline_scores(scores)
    logger.info(f"Baseline initialized with {len(scores)} scores")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force retrain without drift check")
    args = parser.parse_args()

    result = check_and_retrain(force=args.force)
    print(json.dumps(result, indent=2))
