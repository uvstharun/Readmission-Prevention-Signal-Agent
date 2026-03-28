"""
Data loader utilities for the readmission prevention agent.
"""
import pandas as pd
import numpy as np
import pickle
import os
from typing import Optional, Tuple
from src.data.feature_engineering import engineer_features, encode_categoricals, get_feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_processed_features(path: str = "data/processed/features.csv") -> pd.DataFrame:
    """Load processed feature matrix."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed features not found at {path}. Run feature engineering pipeline first.")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} records from {path}")
    return df


def load_raw_data(path: str = "data/synthetic/discharge_cohort.csv") -> pd.DataFrame:
    """Load raw discharge cohort data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found at {path}. Run synthetic data generator first.")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} raw records from {path}")
    return df


def load_preprocessing_artifacts(
    encoders_path: str = "models/encoders.pkl",
    scaler_path: str = "models/scaler.pkl",
) -> Tuple[dict, dict]:
    """Load saved encoders and scaler."""
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler_dict = pickle.load(f)
    return encoders, scaler_dict


def transform_new_patient(patient_dict: dict, encoders_path: str = "models/encoders.pkl",
                           scaler_path: str = "models/scaler.pkl") -> pd.DataFrame:
    """
    Transform a single new patient record into model-ready features.
    Used at inference time by the risk scoring agent.
    """
    df = pd.DataFrame([patient_dict])
    df = engineer_features(df)
    df, _ = encode_categoricals(df, fit=False, encoders=None)

    # Load artifacts
    encoders, scaler_dict = load_preprocessing_artifacts(encoders_path, scaler_path)

    # Re-encode with fitted encoders
    df, _ = encode_categoricals(df, fit=False, encoders=encoders)

    feature_cols = scaler_dict["feature_cols"]
    available_features = [c for c in feature_cols if c in df.columns]

    # Fill missing with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].copy()

    # Impute
    X_imputed = scaler_dict["imputer"].transform(X)
    X_scaled = scaler_dict["scaler"].transform(X_imputed)

    return pd.DataFrame(X_scaled, columns=feature_cols)


def get_train_test_split(
    features_path: str = "data/processed/features.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and split data for model training."""
    from sklearn.model_selection import train_test_split

    df = load_processed_features(features_path)
    feature_cols = [c for c in df.columns if c not in ["patient_id", "readmitted_30_day"]]

    X = df[feature_cols]
    y = df["readmitted_30_day"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Train readmission rate: {y_train.mean():.3f}")
    logger.info(f"Test readmission rate: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test
