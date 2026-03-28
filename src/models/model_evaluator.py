"""
Model evaluation utilities - generates performance metrics, plots, and calibration analysis.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss,
    confusion_matrix, classification_report,
)
from sklearn.calibration import calibration_curve
import json
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_roc_curve(y_true, y_prob, output_path: str = "models/roc_curve.png"):
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Readmission Risk Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")
    return auc


def generate_pr_curve(y_true, y_prob, output_path: str = "models/pr_curve.png"):
    """Generate and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}", color="steelblue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Readmission Risk Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"PR curve saved to {output_path}")
    return ap


def generate_calibration_curve(y_true, y_prob, output_path: str = "models/calibration_curve.png"):
    """Generate and save calibration curve."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10
    )

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model", color="darkorange")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve - Readmission Risk Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Calibration curve saved to {output_path}")


def full_evaluation(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Full model evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    os.makedirs("models", exist_ok=True)
    generate_roc_curve(y_true, y_prob)
    generate_pr_curve(y_true, y_prob)
    generate_calibration_curve(y_true, y_prob)

    return {
        "auc_roc": float(auc),
        "average_precision": float(ap),
        "brier_score": float(brier),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
