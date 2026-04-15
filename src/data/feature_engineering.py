"""
Feature engineering pipeline for readmission risk modeling.
Processes raw discharge data into model-ready feature matrix.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import pickle

# CCSR category mappings for ICD-10 prefixes
CCSR_CATEGORIES = {
    "I50": "Heart Failure",
    "I21": "Acute MI",
    "I22": "Acute MI",
    "I48": "Atrial Fibrillation",
    "I63": "Stroke/Cerebrovascular",
    "I61": "Stroke/Cerebrovascular",
    "J44": "COPD",
    "J43": "COPD",
    "J18": "Pneumonia",
    "J15": "Pneumonia",
    "N18": "Chronic Kidney Disease",
    "E11": "Diabetes",
    "E10": "Diabetes",
    "E13": "Diabetes",
    "A41": "Sepsis",
    "A40": "Sepsis",
    "F32": "Mental Health",
    "F33": "Mental Health",
    "F41": "Mental Health",
    "F10": "Substance Use",
    "F11": "Substance Use",
    "F19": "Substance Use",
    "E43": "Malnutrition",
    "E44": "Malnutrition",
    "C78": "Cancer",
    "C79": "Cancer",
    "C80": "Cancer",
    "K92": "GI Bleed",
    "K57": "GI",
}

HIGH_RISK_DEPARTMENTS = ["Cardiology", "Nephrology", "Oncology"]

CATEGORICAL_COLS = [
    "gender", "race", "ethnicity", "preferred_language",
    "admission_type", "discharge_disposition", "attending_department",
    "insurance_type", "primary_ccsr_category"
]

NUMERIC_COLS = [
    "age", "length_of_stay_days", "charlson_comorbidity_index",
    "prior_admissions_6mo", "prior_ed_visits_6mo", "prior_readmissions_1yr",
    "num_active_medications", "social_support_score",
    "high_risk_medication_flag", "housing_stability_flag",
    "transportation_access_flag", "followup_appointment_scheduled",
    "pcp_assigned_flag", "discharge_instructions_given",
    "num_secondary_diagnoses", "composite_utilization_score",
    "complex_social_risk_score", "care_gap_score",
    "high_utilizer_flag", "polypharmacy_flag",
    "high_risk_department_flag", "age_los_interaction",
    "cci_utilization_interaction",
    # New temporal features
    "discharge_hour", "discharge_weekend", "discharge_season", "discharge_month",
    # New clinical features
    "los_category", "age_group", "diagnosis_diversity",
    "med_cci_interaction", "care_transition_completeness",
    "readmission_risk_index",
]


def map_to_ccsr(icd_code: str) -> str:
    """Map ICD-10 code to CCSR clinical category."""
    if not isinstance(icd_code, str):
        return "Other"
    prefix3 = icd_code[:3]
    prefix4 = icd_code[:4]
    return CCSR_CATEGORIES.get(prefix4, CCSR_CATEGORIES.get(prefix3, "Other"))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full feature engineering pipeline."""
    df = df.copy()

    # Map diagnoses to CCSR categories
    df["primary_ccsr_category"] = df["primary_diagnosis_code"].apply(map_to_ccsr)

    # Count secondary diagnoses
    df["num_secondary_diagnoses"] = df["secondary_diagnosis_codes"].apply(
        lambda x: len(str(x).split("|")) if pd.notna(x) and str(x) != "" else 0
    )

    # Composite utilization score
    df["composite_utilization_score"] = (
        df["prior_admissions_6mo"] * 2
        + df["prior_ed_visits_6mo"]
        + df["prior_readmissions_1yr"] * 3
    )

    # High utilizer flag
    df["high_utilizer_flag"] = (
        (df["prior_admissions_6mo"] >= 2) | (df["prior_ed_visits_6mo"] >= 3)
    ).astype(int)

    # Complex social risk score
    df["complex_social_risk_score"] = (
        (1 - df["housing_stability_flag"])
        + (1 - df["transportation_access_flag"])
        + (7 - df["social_support_score"]) / 7
        + df["insurance_type"].apply(lambda x: 1 if x in ["Medicaid", "Uninsured"] else 0)
    )

    # Polypharmacy flag (5+ medications)
    df["polypharmacy_flag"] = (df["num_active_medications"] >= 5).astype(int)

    # Care gap score
    df["care_gap_score"] = (
        (1 - df["followup_appointment_scheduled"])
        + (1 - df["pcp_assigned_flag"])
        + (1 - df["discharge_instructions_given"])
    )

    # High risk department flag
    df["high_risk_department_flag"] = df["attending_department"].apply(
        lambda x: 1 if x in HIGH_RISK_DEPARTMENTS else 0
    )

    # Interaction features
    df["age_los_interaction"] = df["age"] * df["length_of_stay_days"] / 100
    df["cci_utilization_interaction"] = df["charlson_comorbidity_index"] * df["composite_utilization_score"]

    # --- NEW: Temporal features from discharge_date ---
    if "discharge_date" in df.columns:
        discharge_dt = pd.to_datetime(df["discharge_date"], errors="coerce")
        df["discharge_hour"] = discharge_dt.dt.hour.fillna(12)
        df["discharge_weekend"] = discharge_dt.dt.dayofweek.isin([5, 6]).astype(int)
        df["discharge_month"] = discharge_dt.dt.month.fillna(1)
        df["discharge_season"] = df["discharge_month"].apply(
            lambda m: 0 if m in [12, 1, 2] else (1 if m in [3, 4, 5] else (2 if m in [6, 7, 8] else 3))
        )
    else:
        df["discharge_hour"] = 12
        df["discharge_weekend"] = 0
        df["discharge_month"] = 6
        df["discharge_season"] = 2

    # --- NEW: LOS category (0=short <3d, 1=medium 3-6d, 2=long 7-13d, 3=very long 14+d) ---
    df["los_category"] = pd.cut(
        df["length_of_stay_days"],
        bins=[-1, 2, 6, 13, 999],
        labels=[0, 1, 2, 3]
    ).astype(float).fillna(1)

    # --- NEW: Age group (0=young adult 18-44, 1=middle 45-64, 2=senior 65-79, 3=elderly 80+) ---
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 44, 64, 79, 999],
        labels=[0, 1, 2, 3]
    ).astype(float).fillna(1)

    # --- NEW: Diagnosis diversity (unique CCSR categories across all diagnoses) ---
    def count_unique_ccsr(secondary_codes_str):
        if not secondary_codes_str or str(secondary_codes_str) in ["", "nan"]:
            return 1
        codes = str(secondary_codes_str).split("|")
        categories = set(map_to_ccsr(c) for c in codes if c)
        return len(categories)

    df["diagnosis_diversity"] = df["secondary_diagnosis_codes"].apply(count_unique_ccsr)

    # --- NEW: Medication-CCI interaction ---
    df["med_cci_interaction"] = df["num_active_medications"] * df["charlson_comorbidity_index"]

    # --- NEW: Care transition completeness (0-1 scale) ---
    df["care_transition_completeness"] = (
        df["followup_appointment_scheduled"]
        + df["pcp_assigned_flag"]
        + df["discharge_instructions_given"]
    ) / 3.0

    # --- NEW: Composite readmission risk index ---
    df["readmission_risk_index"] = (
        df["charlson_comorbidity_index"] * 0.25
        + df["composite_utilization_score"] * 0.20
        + df["complex_social_risk_score"] * 0.15
        + df["care_gap_score"] * 0.15
        + (df["length_of_stay_days"] / 10) * 0.10
        + df["high_risk_medication_flag"] * 0.15
    )

    return df


def encode_categoricals(df: pd.DataFrame, fit: bool = True, encoders: dict = None) -> tuple:
    """Label encode categorical columns."""
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = df[col].fillna("Unknown")
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                df[col + "_encoded"] = 0
            else:
                df[col] = df[col].fillna("Unknown")
                df[col + "_encoded"] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    return df, encoders


def get_feature_columns() -> list:
    """Return list of final model feature columns."""
    encoded_cats = [c + "_encoded" for c in CATEGORICAL_COLS]
    return NUMERIC_COLS + encoded_cats


def run_pipeline(
    input_path: str = "data/synthetic/discharge_cohort.csv",
    output_path: str = "data/processed/features.csv",
    encoders_path: str = "models/encoders.pkl",
    scaler_path: str = "models/scaler.pkl",
) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")

    # Engineer features
    df = engineer_features(df)

    # Encode categoricals
    df, encoders = encode_categoricals(df, fit=True)

    # Get feature columns
    feature_cols = get_feature_columns()
    available_features = [c for c in feature_cols if c in df.columns]

    X = df[available_features].copy()

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=available_features)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=available_features)

    # Add target and IDs
    X_scaled_df["patient_id"] = df["patient_id"].values
    if "readmitted_30_day" in df.columns:
        X_scaled_df["readmitted_30_day"] = df["readmitted_30_day"].values

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(encoders_path), exist_ok=True)

    X_scaled_df.to_csv(output_path, index=False)
    print(f"Saved features to {output_path}")

    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)

    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "imputer": imputer, "feature_cols": available_features}, f)

    print(f"Saved encoders and scaler")
    return X_scaled_df


if __name__ == "__main__":
    run_pipeline()
