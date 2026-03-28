"""
Synthetic inpatient discharge data generator for readmission risk modeling.
Generates realistic patient records with appropriate correlations between risk factors and outcomes.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ICD-10 codes by clinical category
DIAGNOSIS_CODES = {
    "heart_failure": ["I50.0", "I50.1", "I50.20", "I50.22", "I50.30", "I50.32", "I50.40", "I50.42"],
    "copd": ["J44.0", "J44.1", "J44.9", "J43.9", "J41.0"],
    "pneumonia": ["J18.9", "J18.0", "J15.9", "J15.0", "J13"],
    "ckd": ["N18.3", "N18.4", "N18.5", "N18.6", "N18.9"],
    "diabetes": ["E11.9", "E11.65", "E11.40", "E10.9", "E13.9"],
    "sepsis": ["A41.9", "A41.51", "A41.01", "A41.02", "A40.0"],
    "afib": ["I48.0", "I48.11", "I48.19", "I48.20", "I48.91"],
    "stroke": ["I63.9", "I63.50", "I61.9", "G45.9"],
    "ami": ["I21.9", "I21.3", "I21.4", "I22.9"],
    "depression": ["F32.9", "F33.0", "F33.1", "F41.1"],
    "substance_use": ["F10.10", "F11.10", "F19.10", "Z87.891"],
    "malnutrition": ["E43", "E44.0", "E44.1", "E46"],
    "falls_trauma": ["W19.XXXA", "S09.90XA", "S00.90XA"],
    "gi": ["K92.1", "K57.30", "K25.9", "K26.9"],
    "cancer": ["C78.00", "C79.89", "Z85.3", "C80.1"],
}

PROCEDURE_CODES = [
    "99213", "99214", "99215", "93306", "93307", "71046", "80053",
    "36415", "93000", "93005", "85025", "80048", "82565", "84484"
]

DEPARTMENTS = [
    "Medicine", "Cardiology", "Pulmonology", "Nephrology",
    "Oncology", "Neurology", "Orthopedics", "General Surgery"
]

DISCHARGE_DISPOSITIONS = [
    "Home", "Home with Home Health", "SNF", "Rehab Facility",
    "Long Term Care", "Hospice", "AMA"
]

ADMISSION_TYPES = ["Emergency", "Urgent", "Elective", "Observation"]

INSURANCE_TYPES = ["Medicare", "Medicaid", "Commercial", "Medicare Advantage", "Uninsured"]

RACES = ["White", "Black or African American", "Asian", "Hispanic or Latino", "Other"]

LANGUAGES = ["English", "Spanish", "Mandarin", "Vietnamese", "Other"]

ZIP_CODES = [str(x) for x in range(10001, 10100)]


def calculate_readmission_probability(row_data: dict) -> float:
    """Calculate readmission probability based on clinical risk factors."""
    base_prob = 0.10

    # Age factor
    age = row_data["age"]
    if age >= 80:
        base_prob += 0.12
    elif age >= 70:
        base_prob += 0.08
    elif age >= 60:
        base_prob += 0.04

    # Length of stay
    los = row_data["length_of_stay_days"]
    if los >= 10:
        base_prob += 0.10
    elif los >= 7:
        base_prob += 0.06
    elif los >= 4:
        base_prob += 0.03

    # Prior utilization
    prior_admissions = row_data["prior_admissions_6mo"]
    if prior_admissions >= 3:
        base_prob += 0.15
    elif prior_admissions >= 1:
        base_prob += 0.08

    prior_ed = row_data["prior_ed_visits_6mo"]
    if prior_ed >= 3:
        base_prob += 0.08
    elif prior_ed >= 1:
        base_prob += 0.04

    # Charlson score
    cci = row_data["charlson_comorbidity_index"]
    if cci >= 6:
        base_prob += 0.15
    elif cci >= 4:
        base_prob += 0.10
    elif cci >= 2:
        base_prob += 0.05

    # High risk medications
    if row_data["high_risk_medication_flag"] == 1:
        base_prob += 0.06

    # Polypharmacy
    n_meds = row_data["num_active_medications"]
    if n_meds >= 10:
        base_prob += 0.06
    elif n_meds >= 6:
        base_prob += 0.03

    # Social determinants
    if row_data["housing_stability_flag"] == 0:
        base_prob += 0.08
    if row_data["transportation_access_flag"] == 0:
        base_prob += 0.05
    social_support = row_data["social_support_score"]
    if social_support <= 2:
        base_prob += 0.06
    elif social_support <= 4:
        base_prob += 0.03

    # Insurance
    if row_data["insurance_type"] in ["Medicaid", "Uninsured"]:
        base_prob += 0.06

    # Care transition factors
    if row_data["followup_appointment_scheduled"] == 0:
        base_prob += 0.08
    if row_data["pcp_assigned_flag"] == 0:
        base_prob += 0.05
    if row_data["discharge_instructions_given"] == 0:
        base_prob += 0.04

    # Discharge disposition
    if row_data["discharge_disposition"] in ["AMA", "Home"]:
        base_prob += 0.05
    elif row_data["discharge_disposition"] == "SNF":
        base_prob -= 0.03

    # Diagnosis
    primary_diag = row_data["primary_diagnosis_code"]
    high_risk_diags = (
        DIAGNOSIS_CODES["heart_failure"]
        + DIAGNOSIS_CODES["sepsis"]
        + DIAGNOSIS_CODES["ckd"]
        + DIAGNOSIS_CODES["copd"]
    )
    if primary_diag in high_risk_diags:
        base_prob += 0.08

    return min(0.95, max(0.02, base_prob))


def generate_secondary_diagnoses(primary_code: str, cci: int) -> list:
    """Generate realistic secondary diagnoses based on primary and CCI score."""
    secondary = []
    all_codes = []
    for codes in DIAGNOSIS_CODES.values():
        all_codes.extend(codes)

    num_secondary = min(cci + 1, 10)
    num_secondary = max(0, np.random.poisson(num_secondary))
    num_secondary = min(num_secondary, 10)

    potential = [c for c in all_codes if c != primary_code]
    if potential:
        secondary = np.random.choice(potential, size=min(num_secondary, len(potential)), replace=False).tolist()

    return secondary


def compute_charlson_index(diagnoses: list) -> int:
    """Estimate CCI from diagnosis list."""
    score = 0
    combined = " ".join(diagnoses)

    if any(d in combined for d in ["I50", "I21", "I22"]):
        score += 1
    if any(d in combined for d in ["I63", "I61", "G45"]):
        score += 1
    if any(d in combined for d in ["E11", "E10", "E13"]):
        score += 1
    if any(d in combined for d in ["N18.3", "N18.4"]):
        score += 1
    if any(d in combined for d in ["N18.5", "N18.6"]):
        score += 2
    if any(d in combined for d in ["J44", "J43"]):
        score += 1
    if any(d in combined for d in ["C78", "C79", "C80"]):
        score += 2
    if any(d in combined for d in ["I48"]):
        score += 1

    return score


def generate_synthetic_dataset(n_records: int = 10000) -> pd.DataFrame:
    """Generate synthetic patient discharge dataset."""
    records = []
    base_date = datetime(2024, 1, 1)

    for i in range(n_records):
        patient_id = f"PT{100000 + i}"
        age = int(np.random.choice(
            range(18, 95),
            p=None
        ))
        # Weight age toward elderly
        age = int(np.clip(np.random.normal(65, 18), 18, 94))

        gender = np.random.choice(["Male", "Female", "Non-binary"], p=[0.48, 0.50, 0.02])
        race = np.random.choice(RACES, p=[0.55, 0.18, 0.06, 0.16, 0.05])
        ethnicity = np.random.choice(["Hispanic", "Non-Hispanic"], p=[0.18, 0.82])
        language = np.random.choice(LANGUAGES, p=[0.72, 0.14, 0.05, 0.04, 0.05])
        zip_code = np.random.choice(ZIP_CODES)

        # Encounter details
        admission_offset = random.randint(0, 365)
        admission_date = base_date + timedelta(days=admission_offset)
        los = max(1, int(np.random.lognormal(1.5, 0.8)))
        los = min(los, 45)
        discharge_date = admission_date + timedelta(days=los)
        admission_type = np.random.choice(ADMISSION_TYPES, p=[0.55, 0.20, 0.15, 0.10])
        discharge_disposition = np.random.choice(
            DISCHARGE_DISPOSITIONS,
            p=[0.40, 0.25, 0.15, 0.08, 0.05, 0.04, 0.03]
        )
        department = np.random.choice(DEPARTMENTS)

        # Diagnoses
        all_codes_flat = []
        for codes in DIAGNOSIS_CODES.values():
            all_codes_flat.extend(codes)

        primary_diag = np.random.choice(all_codes_flat)
        cci_base = max(0, int(np.random.normal(2.5, 2.0)))
        secondary_diags = generate_secondary_diagnoses(primary_diag, cci_base)
        all_diags = [primary_diag] + secondary_diags
        cci = compute_charlson_index(all_diags)
        procedure_code = np.random.choice(PROCEDURE_CODES)

        # Utilization history
        prior_admissions_6mo = max(0, int(np.random.negative_binomial(1, 0.7)))
        prior_ed_6mo = max(0, int(np.random.negative_binomial(1, 0.6)))
        prior_readmissions_1yr = max(0, int(np.random.binomial(3, 0.15)))

        # Medications
        n_meds = max(0, int(np.random.normal(7, 3.5)))
        n_meds = min(n_meds, 25)
        high_risk_med = 1 if np.random.random() < 0.35 else 0

        # Social determinants
        insurance = np.random.choice(INSURANCE_TYPES, p=[0.35, 0.20, 0.30, 0.12, 0.03])
        housing = 1 if np.random.random() < 0.85 else 0
        transport = 1 if np.random.random() < 0.78 else 0
        social_support = int(np.random.choice([1, 2, 3, 4, 5, 6, 7], p=[0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.10]))

        # Care transition factors
        followup = 1 if np.random.random() < 0.65 else 0
        pcp_assigned = 1 if np.random.random() < 0.72 else 0
        dc_instructions = 1 if np.random.random() < 0.80 else 0

        row_data = {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "race": race,
            "ethnicity": ethnicity,
            "preferred_language": language,
            "zip_code": zip_code,
            "admission_date": admission_date.strftime("%Y-%m-%d"),
            "discharge_date": discharge_date.strftime("%Y-%m-%d"),
            "length_of_stay_days": los,
            "admission_type": admission_type,
            "discharge_disposition": discharge_disposition,
            "attending_department": department,
            "primary_diagnosis_code": primary_diag,
            "secondary_diagnosis_codes": "|".join(secondary_diags),
            "primary_procedure_code": procedure_code,
            "charlson_comorbidity_index": cci,
            "prior_admissions_6mo": prior_admissions_6mo,
            "prior_ed_visits_6mo": prior_ed_6mo,
            "prior_readmissions_1yr": prior_readmissions_1yr,
            "num_active_medications": n_meds,
            "high_risk_medication_flag": high_risk_med,
            "insurance_type": insurance,
            "housing_stability_flag": housing,
            "transportation_access_flag": transport,
            "social_support_score": social_support,
            "followup_appointment_scheduled": followup,
            "pcp_assigned_flag": pcp_assigned,
            "discharge_instructions_given": dc_instructions,
        }

        # Readmission outcome
        prob = calculate_readmission_probability(row_data)
        readmitted = int(np.random.random() < prob)

        row_data["readmission_probability_true"] = round(prob, 4)
        row_data["readmitted_30_day"] = readmitted

        records.append(row_data)

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} records")
    print(f"Readmission rate: {df['readmitted_30_day'].mean():.3f}")
    return df


if __name__ == "__main__":
    os.makedirs("data/synthetic", exist_ok=True)
    df = generate_synthetic_dataset(10000)
    output_path = "data/synthetic/discharge_cohort.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(df.head())
    print(df.dtypes)
