"""
Model training pipeline for readmission risk prediction.
Trains XGBoost, LightGBM, Logistic Regression, MLP, and a Stacking Ensemble.
Selects the best model and saves with SHAP explainability.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import shap
import warnings

warnings.filterwarnings("ignore")

from src.data.data_loader import get_train_test_split
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_base_models(y_train):
    """Build all base model definitions."""
    pos_weight = sum(y_train == 0) / max(sum(y_train == 1), 1)

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.1, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
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
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate="adaptive",
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    }


def _build_ensemble(y_train):
    """
    Build a stacking ensemble:
    Base learners: XGBoost + LightGBM + MLP
    Meta-learner: Logistic Regression
    """
    pos_weight = sum(y_train == 0) / max(sum(y_train == 1), 1)

    estimators = [
        ("xgboost", XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, random_state=42,
            eval_metric="logloss", verbosity=0,
        )),
        ("lightgbm", LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, verbose=-1,
        )),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", alpha=0.001, max_iter=150,
            random_state=42, early_stopping=True,
        )),
    ]

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=42),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


def train_models(X_train, y_train, X_test, y_test) -> dict:
    """Train all models and return evaluation results."""

    # SMOTE for class imbalance (used for LR and MLP)
    logger.info("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

    base_models = _build_base_models(y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # Train base models
    for name, model in base_models.items():
        logger.info(f"Training {name}...")

        # LR and MLP benefit from SMOTE; tree models handle imbalance natively
        train_X = X_resampled if name in ["logistic_regression", "mlp"] else X_train
        train_y = y_resampled if name in ["logistic_regression", "mlp"] else y_train

        model.fit(train_X, train_y)

        cv_scores = cross_val_score(model, train_X, train_y, cv=cv, scoring="roc_auc", n_jobs=-1)
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
        logger.info(f"{name}: CV AUC={cv_scores.mean():.4f}±{cv_scores.std():.4f} | "
                    f"Test AUC={auc_roc:.4f} | AP={avg_precision:.4f} | Brier={brier:.4f}")

    # Train stacking ensemble
    logger.info("Training stacking ensemble (XGBoost + LightGBM + MLP)...")
    ensemble = _build_ensemble(y_train)
    ensemble.fit(X_train, y_train)

    y_prob_ens = ensemble.predict_proba(X_test)[:, 1]
    auc_ens = roc_auc_score(y_test, y_prob_ens)
    ap_ens = average_precision_score(y_test, y_prob_ens)
    brier_ens = brier_score_loss(y_test, y_prob_ens)

    results["stacking_ensemble"] = {
        "model": ensemble,
        "cv_auc_mean": auc_ens,
        "cv_auc_std": 0.0,
        "test_auc_roc": float(auc_ens),
        "test_avg_precision": float(ap_ens),
        "test_brier_score": float(brier_ens),
    }
    logger.info(f"stacking_ensemble: Test AUC={auc_ens:.4f} | AP={ap_ens:.4f} | Brier={brier_ens:.4f}")

    return results


def select_best_model(results: dict) -> tuple:
    """Select best model by test AUC-ROC."""
    best_name = max(results, key=lambda k: results[k]["test_auc_roc"])
    logger.info(f"Best model: {best_name} (AUC={results[best_name]['test_auc_roc']:.4f})")
    return best_name, results[best_name]["model"]


def build_shap_explainer(model, X_train, model_name: str):
    """Build SHAP explainer. Uses TreeExplainer for tree models, KernelExplainer for others."""
    logger.info("Building SHAP explainer...")

    if model_name in ["xgboost", "lightgbm"]:
        return shap.TreeExplainer(model)

    if model_name == "logistic_regression":
        return shap.LinearExplainer(model, X_train)

    if model_name == "mlp":
        # KernelExplainer on a small background sample for speed
        background = shap.sample(X_train, 100)
        return shap.KernelExplainer(model.predict_proba, background)

    if model_name == "stacking_ensemble":
        # Use the XGBoost base learner's explainer as proxy
        xgb_model = dict(model.named_estimators_).get("xgboost")
        if xgb_model:
            return shap.TreeExplainer(xgb_model)
        background = shap.sample(X_train, 100)
        return shap.KernelExplainer(model.predict_proba, background)

    return shap.LinearExplainer(model, X_train)


def save_artifacts(
    model,
    explainer,
    feature_names: list,
    evaluation_report: dict,
    model_path: str = "models/readmission_risk_model.pkl",
    explainer_path: str = "models/shap_explainer.pkl",
    report_path: str = "models/evaluation_report.json",
):
    """Save model, SHAP explainer, and evaluation report."""
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
    logger.info("=" * 60)
    logger.info("Starting enhanced model training pipeline")
    logger.info("Models: LR | XGBoost | LightGBM | MLP | Stacking Ensemble")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = get_train_test_split()
    feature_names = list(X_train.columns)
    logger.info(f"Features: {len(feature_names)} | Train: {len(X_train)} | Test: {len(X_test)}")

    results = train_models(X_train, y_train, X_test, y_test)
    best_name, best_model = select_best_model(results)
    explainer = build_shap_explainer(best_model, X_train, best_name)

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

    logger.info("=" * 60)
    logger.info(f"Training complete | Best: {best_name} | AUC: {results[best_name]['test_auc_roc']:.4f}")
    logger.info("=" * 60)
    return evaluation_report


if __name__ == "__main__":
    report = run_training_pipeline()
    print(json.dumps(report, indent=2))
