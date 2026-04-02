"""
MIMIC-IV discharge cohort builder.
Constructs the index cohort with 30-day readmission labels.

Cohort criteria:
  - Adult inpatients (age >= 18)
  - Discharged alive (exclude in-hospital deaths)
  - Index admission: first or any admission per patient
  - Readmission: any unplanned admission within 30 days of discharge
  - Exclude elective readmissions (planned)
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Discharge locations that indicate planned follow-up (not unplanned readmissions)
PLANNED_ADMISSION_TYPES = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}

# Discharge locations mapped to our standard categories
DISCHARGE_LOCATION_MAP = {
    "HOME": "Home",
    "HOME HEALTH CARE": "Home with Home Health",
    "SKILLED NURSING FACILITY": "SNF",
    "REHAB": "Rehab Facility",
    "CHRONIC/LONG TERM ACUTE CARE": "Long Term Care",
    "HOSPICE": "Hospice",
    "AGAINST ADVICE": "AMA",
    "OTHER FACILITY": "SNF",
    "ACUTE HOSPITAL": "SNF",
}

INSURANCE_MAP = {
    "Medicare": "Medicare",
    "Medicaid": "Medicaid",
    "Other": "Commercial",
    "No charge": "Uninsured",
}


def build_cohort(
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    max_patients: int = None,
) -> pd.DataFrame:
    """
    Build the discharge cohort from MIMIC-IV admissions and patients tables.
    Returns one row per admission with 30-day readmission label.
    """
    logger.info("Building discharge cohort...")

    # Merge patients onto admissions
    adm = admissions.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year", "dod"]],
        on="subject_id",
        how="left",
    )

    # Compute approximate age at admission
    adm["age"] = adm["anchor_age"] + (
        adm["admittime"].dt.year - adm["anchor_year"]
    ).clip(lower=0)

    # Filter: adults only
    adm = adm[adm["age"] >= 18].copy()

    # Filter: exclude in-hospital deaths
    adm = adm[adm["hospital_expire_flag"] == 0].copy()

    # Filter: must have valid discharge time
    adm = adm[adm["dischtime"].notna()].copy()

    # Sort by patient and admission time
    adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    if max_patients:
        subjects = adm["subject_id"].unique()[:max_patients]
        adm = adm[adm["subject_id"].isin(subjects)].copy()

    logger.info(f"Cohort after filters: {len(adm)} admissions, {adm['subject_id'].nunique()} patients")

    # Compute 30-day readmission label
    adm = _label_readmissions(adm)

    # Map fields to our standard schema
    adm = _map_fields(adm)

    logger.info(f"Final cohort: {len(adm)} admissions | readmission rate: {adm['readmitted_30_day'].mean():.3f}")
    return adm


def _label_readmissions(adm: pd.DataFrame) -> pd.DataFrame:
    """Label each admission with 30-day unplanned readmission."""
    adm = adm.copy()
    adm["readmitted_30_day"] = 0

    # For each patient, check if next admission is within 30 days
    adm_sorted = adm.sort_values(["subject_id", "admittime"]).copy()
    adm_sorted["next_admittime"] = adm_sorted.groupby("subject_id")["admittime"].shift(-1)
    adm_sorted["next_admission_type"] = adm_sorted.groupby("subject_id")["admission_type"].shift(-1)

    # 30-day window
    adm_sorted["days_to_next"] = (
        adm_sorted["next_admittime"] - adm_sorted["dischtime"]
    ).dt.total_seconds() / 86400

    # Readmission: next admission within 30 days AND not elective
    readmit_mask = (
        (adm_sorted["days_to_next"] >= 0) &
        (adm_sorted["days_to_next"] <= 30) &
        (~adm_sorted["next_admission_type"].isin(PLANNED_ADMISSION_TYPES))
    )
    adm_sorted.loc[readmit_mask, "readmitted_30_day"] = 1

    return adm_sorted


def _map_fields(adm: pd.DataFrame) -> pd.DataFrame:
    """Map MIMIC-IV fields to our standard schema."""
    adm = adm.copy()

    # Patient ID
    adm["patient_id"] = "MIMIC_" + adm["hadm_id"].astype(str)

    # Dates
    adm["admission_date"] = adm["admittime"].dt.strftime("%Y-%m-%d")
    adm["discharge_date"] = adm["dischtime"].dt.strftime("%Y-%m-%d")
    adm["length_of_stay_days"] = (
        (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400
    ).round(0).astype(int).clip(lower=1)

    # Demographics
    adm["gender"] = adm["gender"].map({"M": "Male", "F": "Female"}).fillna("Unknown")
    adm["race"] = adm["race"].apply(_map_race)
    adm["ethnicity"] = adm["race"].apply(
        lambda x: "Hispanic" if "HISPANIC" in str(x).upper() else "Non-Hispanic"
    )
    adm["preferred_language"] = adm.get("language", "English").fillna("English")
    adm["zip_code"] = "00000"  # Not available in MIMIC

    # Encounter
    adm["admission_type"] = adm["admission_type"].apply(_map_admission_type)
    adm["discharge_disposition"] = adm["discharge_location"].apply(_map_discharge_location)
    adm["attending_department"] = "Medicine"  # Will be enriched later

    # Insurance
    adm["insurance_type"] = adm["insurance"].apply(_map_insurance)

    # Fill remaining required columns with defaults (enriched by feature extractor)
    for col in ["charlson_comorbidity_index", "prior_admissions_6mo", "prior_ed_visits_6mo",
                "prior_readmissions_1yr", "num_active_medications", "high_risk_medication_flag",
                "housing_stability_flag", "transportation_access_flag", "social_support_score",
                "followup_appointment_scheduled", "pcp_assigned_flag", "discharge_instructions_given"]:
        if col not in adm.columns:
            adm[col] = 0

    adm["primary_diagnosis_code"] = ""
    adm["secondary_diagnosis_codes"] = ""

    return adm


def _map_race(race_str: str) -> str:
    race_str = str(race_str).upper()
    if "WHITE" in race_str:
        return "White"
    if "BLACK" in race_str or "AFRICAN" in race_str:
        return "Black or African American"
    if "ASIAN" in race_str:
        return "Asian"
    if "HISPANIC" in race_str or "LATINO" in race_str:
        return "Hispanic or Latino"
    return "Other"


def _map_admission_type(t: str) -> str:
    t = str(t).upper()
    if "EMERGENCY" in t or "EU OBSERVATION" in t:
        return "Emergency"
    if "URGENT" in t:
        return "Urgent"
    if "ELECTIVE" in t:
        return "Elective"
    return "Urgent"


def _map_discharge_location(loc: str) -> str:
    loc = str(loc).upper()
    for key, val in DISCHARGE_LOCATION_MAP.items():
        if key in loc:
            return val
    return "Home"


def _map_insurance(ins: str) -> str:
    return INSURANCE_MAP.get(str(ins), "Commercial")
