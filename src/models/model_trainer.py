"""
Model training pipeline for readmission risk prediction.
Trains, evaluates, and saves the best model with SHAP explainability.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import shap
import warnings

warnings.filterwarnings("ignore")

from src.data.data_loader import get_train_test_split
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_models(X_train, y_train, X_test, y_test) -> dict:
    """Train and compare multiple models."""

    # Handle class imbalance with SMOTE
    logger.info("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {y_resampled.value_counts().to_dict()}")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.1, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        logger.info(f"Training {name}...")

        if name == "logistic_regression":
            train_X, train_y = X_resampled, y_resampled
        else:
            train_X, train_y = X_train, y_train

        model.fit(train_X, train_y)

        # CV score
        cv_scores = cross_val_score(model, train_X, train_y, cv=cv, scoring="roc_auc", n_jobs=-1)

        # Test metrics
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        results[name] = {
            "model": model,
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "test_auc_roc": float(auc_roc),
            "test_avg_precision": float(avg_precision),
            "test_brier_score": float(brier),
        }

        logger.info(f"{name}: CV AUC={cv_scores.mean():.4f}±{cv_scores.std():.4f}, "
                    f"Test AUC={auc_roc:.4f}, AP={avg_precision:.4f}, Brier={brier:.4f}")

    return results


def select_best_model(results: dict) -> tuple:
    """Select best model based on test AUC-ROC."""
    best_name = max(results, key=lambda k: results[k]["test_auc_roc"])
    logger.info(f"Best model: {best_name} (AUC={results[best_name]['test_auc_roc']:.4f})")
    return best_name, results[best_name]["model"]


def build_shap_explainer(model, X_train, model_name: str):
    """Build SHAP explainer for the selected model."""
    logger.info("Building SHAP explainer...")

    if model_name in ["xgboost", "lightgbm"]:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_train)

    return explainer


def save_artifacts(
    model,
    explainer,
    feature_names: list,
    evaluation_report: dict,
    model_path: str = "models/readmission_risk_model.pkl",
    explainer_path: str = "models/shap_explainer.pkl",
    report_path: str = "models/evaluation_report.json",
):
    """Save model, explainer, and evaluation report."""
    os.makedirs("models", exist_ok=True)

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "threshold_high": config.RISK_THRESHOLD_HIGH,
        "threshold_moderate": config.RISK_THRESHOLD_MODERATE,
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info(f"Saved model to {model_path}")

    with open(explainer_path, "wb") as f:
        pickle.dump(explainer, f)
    logger.info(f"Saved SHAP explainer to {explainer_path}")

    with open(report_path, "w") as f:
        json.dump(evaluation_report, f, indent=2)
    logger.info(f"Saved evaluation report to {report_path}")


def run_training_pipeline():
    """Full model training pipeline."""
    logger.info("Starting model training pipeline")

    # Load data
    X_train, X_test, y_train, y_test = get_train_test_split()
    feature_names = list(X_train.columns)

    # Train models
    results = train_models(X_train, y_train, X_test, y_test)

    # Select best
    best_name, best_model = select_best_model(results)

    # Build SHAP explainer
    explainer = build_shap_explainer(best_model, X_train, best_name)

    # Build evaluation report
    evaluation_report = {
        "best_model": best_name,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "all_models": {
            name: {k: v for k, v in metrics.items() if k != "model"}
            for name, metrics in results.items()
        },
        "best_model_metrics": {
            k: v for k, v in results[best_name].items() if k != "model"
        },
    }

    save_artifacts(best_model, explainer, feature_names, evaluation_report)
    logger.info("Training pipeline complete")
    return evaluation_report


if __name__ == "__main__":
    report = run_training_pipeline()
    print(json.dumps(report, indent=2))
