"""
Full MIMIC-IV processing pipeline.
Orchestrates loading, cohort building, and feature extraction.
"""
import os
import pandas as pd
from src.data.mimic.mimic_loader import (
    validate_mimic_files, load_admissions, load_patients,
    load_diagnoses, load_prescriptions, load_labevents,
)
from src.data.mimic.cohort_builder import build_cohort
from src.data.mimic.feature_extractor import (
    extract_diagnoses_features, extract_medication_features,
    extract_utilization_features, extract_lab_features,
)
from src.data.feature_engineering import run_pipeline as run_feature_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_mimic_pipeline(
    max_patients: int = None,
    lab_nrows: int = 5_000_000,
    output_raw: str = "data/processed/mimic_cohort_raw.csv",
    output_features: str = "data/processed/features.csv",
):
    """
    Full MIMIC-IV pipeline from raw files to model-ready feature matrix.

    Args:
        max_patients: Limit to N patients (useful for testing). None = all patients.
        lab_nrows: Max lab rows to load (labevents.csv is very large).
        output_raw: Where to save the raw merged cohort before feature engineering.
        output_features: Final feature matrix path.
    """
    logger.info("=" * 60)
    logger.info("Starting MIMIC-IV pipeline")
    logger.info("=" * 60)

    # Step 1: Validate files
    validate_mimic_files()

    # Step 2: Load raw tables
    logger.info("Loading MIMIC-IV tables...")
    admissions = load_admissions()
    patients = load_patients()
    diagnoses = load_diagnoses()
    prescriptions = load_prescriptions()
    labevents = load_labevents(nrows=lab_nrows)

    # Step 3: Build cohort with readmission labels
    cohort = build_cohort(admissions, patients, max_patients=max_patients)

    # Step 4: Add subject_id back for utilization calc
    cohort = cohort.merge(
        admissions[["hadm_id", "subject_id"]],
        on="hadm_id", how="left"
    )

    # Step 5: Enrich with clinical features
    cohort = extract_diagnoses_features(diagnoses, cohort)
    cohort = extract_medication_features(prescriptions, cohort)
    cohort = extract_utilization_features(admissions, cohort)
    cohort = extract_lab_features(labevents, cohort)

    # Step 6: Fill SDOH fields (not available in MIMIC — set sensible defaults)
    # In production, these would come from census/zip code enrichment
    cohort["housing_stability_flag"] = 1
    cohort["transportation_access_flag"] = 1
    cohort["social_support_score"] = 4
    cohort["followup_appointment_scheduled"] = 0
    cohort["pcp_assigned_flag"] = 1
    cohort["discharge_instructions_given"] = 1
    cohort["zip_code"] = "00000"

    # Step 7: Save raw cohort
    os.makedirs(os.path.dirname(output_raw), exist_ok=True)
    cohort.to_csv(output_raw, index=False)
    logger.info(f"Saved raw MIMIC cohort: {len(cohort)} rows → {output_raw}")
    logger.info(f"Readmission rate: {cohort['readmitted_30_day'].mean():.3f}")

    # Step 8: Run standard feature engineering pipeline on MIMIC cohort
    logger.info("Running feature engineering on MIMIC cohort...")
    run_feature_pipeline(
        input_path=output_raw,
        output_path=output_features,
    )

    logger.info("MIMIC-IV pipeline complete")
    return cohort


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Limit number of patients (for testing)")
    parser.add_argument("--lab-nrows", type=int, default=5_000_000)
    args = parser.parse_args()

    run_mimic_pipeline(max_patients=args.max_patients, lab_nrows=args.lab_nrows)
