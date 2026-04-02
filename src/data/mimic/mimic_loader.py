"""
MIMIC-IV data loader.
Loads raw MIMIC-IV CSV files from data/mimic/ and validates their presence.

Required MIMIC-IV files (place in data/mimic/hosp/):
  - admissions.csv
  - patients.csv
  - diagnoses_icd.csv
  - prescriptions.csv
  - labevents.csv
  - d_labitems.csv

Optional:
  - procedures_icd.csv
  - drgcodes.csv
"""
import os
import pandas as pd
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

MIMIC_BASE = "data/mimic/hosp"

REQUIRED_FILES = [
    "admissions.csv",
    "patients.csv",
    "diagnoses_icd.csv",
    "prescriptions.csv",
]

OPTIONAL_FILES = [
    "labevents.csv",
    "d_labitems.csv",
    "procedures_icd.csv",
    "drgcodes.csv",
]


def check_mimic_files() -> dict:
    """Check which MIMIC files are present."""
    status = {}
    for f in REQUIRED_FILES:
        path = os.path.join(MIMIC_BASE, f)
        status[f] = {"present": os.path.exists(path), "required": True, "path": path}
    for f in OPTIONAL_FILES:
        path = os.path.join(MIMIC_BASE, f)
        status[f] = {"present": os.path.exists(path), "required": False, "path": path}
    return status


def validate_mimic_files():
    """Raise error if required MIMIC files are missing."""
    status = check_mimic_files()
    missing = [f for f, s in status.items() if s["required"] and not s["present"]]
    if missing:
        raise FileNotFoundError(
            f"Missing required MIMIC-IV files in {MIMIC_BASE}/:\n"
            + "\n".join(f"  - {f}" for f in missing)
            + "\n\nDownload MIMIC-IV from https://physionet.org/content/mimiciv/"
        )
    logger.info("All required MIMIC-IV files found")


def load_admissions(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load admissions table."""
    path = os.path.join(MIMIC_BASE, "admissions.csv")
    df = pd.read_csv(path, nrows=nrows, parse_dates=["admittime", "dischtime", "deathtime"])
    logger.info(f"Loaded admissions: {len(df)} rows")
    return df


def load_patients(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load patients table."""
    path = os.path.join(MIMIC_BASE, "patients.csv")
    df = pd.read_csv(path, nrows=nrows)
    logger.info(f"Loaded patients: {len(df)} rows")
    return df


def load_diagnoses(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load diagnoses_icd table."""
    path = os.path.join(MIMIC_BASE, "diagnoses_icd.csv")
    df = pd.read_csv(path, nrows=nrows, dtype={"icd_code": str})
    logger.info(f"Loaded diagnoses: {len(df)} rows")
    return df


def load_prescriptions(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load prescriptions table."""
    path = os.path.join(MIMIC_BASE, "prescriptions.csv")
    df = pd.read_csv(path, nrows=nrows, parse_dates=["starttime", "stoptime"])
    logger.info(f"Loaded prescriptions: {len(df)} rows")
    return df


def load_labevents(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load labevents table (large file — use nrows for testing)."""
    path = os.path.join(MIMIC_BASE, "labevents.csv")
    if not os.path.exists(path):
        logger.warning("labevents.csv not found — lab features will be skipped")
        return pd.DataFrame()
    df = pd.read_csv(path, nrows=nrows, parse_dates=["charttime"],
                     usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "flag"])
    logger.info(f"Loaded labevents: {len(df)} rows")
    return df


def load_d_labitems() -> pd.DataFrame:
    """Load lab item dictionary."""
    path = os.path.join(MIMIC_BASE, "d_labitems.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)
