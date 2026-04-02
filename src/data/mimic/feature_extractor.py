"""
MIMIC-IV feature extractor.
Enriches the base cohort with clinical features from diagnoses, prescriptions, and labs.
"""
import pandas as pd
import numpy as np
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Charlson Comorbidity Index ICD-10 mapping
CCI_ICD10_MAP = {
    "MI":             {"prefixes": ["I21", "I22", "I252"], "points": 1},
    "CHF":            {"prefixes": ["I099", "I110", "I130", "I132", "I255", "I420", "I425",
                                    "I426", "I427", "I428", "I429", "I43", "I50", "P290"], "points": 1},
    "PVD":            {"prefixes": ["I70", "I71", "I731", "I738", "I739", "I771", "I790",
                                    "I792", "K551", "K558", "K559", "Z958", "Z959"], "points": 1},
    "CVD":            {"prefixes": ["G45", "G46", "H340", "I60", "I61", "I62", "I63",
                                    "I64", "I65", "I66", "I67", "I68", "I69"], "points": 1},
    "Dementia":       {"prefixes": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"], "points": 1},
    "COPD":           {"prefixes": ["J40", "J41", "J42", "J43", "J44", "J45", "J46",
                                    "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67"], "points": 1},
    "Rheumatologic":  {"prefixes": ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"], "points": 1},
    "PUD":            {"prefixes": ["K25", "K26", "K27", "K28"], "points": 1},
    "Mild Liver":     {"prefixes": ["B18", "K700", "K701", "K702", "K703", "K709", "K713",
                                    "K714", "K715", "K717", "K73", "K74", "K760", "K762",
                                    "K763", "K764", "K768", "K769", "Z944"], "points": 1},
    "DM No Comp":     {"prefixes": ["E100", "E101", "E106", "E108", "E109", "E110", "E111",
                                    "E116", "E118", "E119", "E120", "E121", "E126", "E128",
                                    "E129", "E130", "E131", "E136", "E138", "E139"], "points": 1},
    "DM Comp":        {"prefixes": ["E102", "E103", "E104", "E105", "E107", "E112", "E113",
                                    "E114", "E115", "E117", "E132", "E133", "E134", "E135", "E137"], "points": 2},
    "Hemiplegia":     {"prefixes": ["G041", "G114", "G801", "G802", "G81", "G82",
                                    "G830", "G831", "G832", "G833", "G834", "G839"], "points": 2},
    "Renal":          {"prefixes": ["I120", "I131", "N032", "N033", "N034", "N035",
                                    "N036", "N037", "N052", "N053", "N054", "N055",
                                    "N056", "N057", "N18", "N19", "N250", "Z490",
                                    "Z491", "Z492", "Z940", "Z992"], "points": 2},
    "Malignancy":     {"prefixes": ["C0", "C1", "C2", "C30", "C31", "C32", "C33", "C34",
                                    "C37", "C38", "C39", "C40", "C41", "C43", "C45", "C46",
                                    "C47", "C48", "C49", "C50", "C51", "C52", "C53", "C54",
                                    "C55", "C56", "C57", "C58", "C60", "C61", "C62", "C63",
                                    "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71",
                                    "C72", "C73", "C74", "C75", "C76", "C81", "C82", "C83",
                                    "C84", "C85", "C88", "C90", "C91", "C92", "C93", "C94",
                                    "C95", "C96", "C97"], "points": 2},
    "Severe Liver":   {"prefixes": ["I850", "I859", "I864", "I982", "K704", "K711",
                                    "K721", "K729", "K765", "K766", "K767"], "points": 3},
    "Metastatic":     {"prefixes": ["C77", "C78", "C79", "C80"], "points": 6},
    "AIDS":           {"prefixes": ["B20", "B21", "B22", "B24"], "points": 6},
}

# High-risk medication keywords
HIGH_RISK_MED_KEYWORDS = [
    "warfarin", "heparin", "enoxaparin", "apixaban", "rivaroxaban", "dabigatran",
    "insulin", "nph insulin", "glargine", "detemir",
    "morphine", "oxycodone", "hydromorphone", "fentanyl", "methadone", "codeine",
    "digoxin", "amiodarone", "lithium", "methotrexate",
]

# Key lab item IDs in MIMIC-IV
KEY_LAB_ITEMS = {
    50912: "creatinine",
    50931: "glucose",
    50971: "potassium",
    50983: "sodium",
    51222: "hemoglobin",
    51301: "wbc",
    51265: "platelets",
    50882: "bicarbonate",
    50868: "anion_gap",
    50820: "ph",
}


def compute_cci_from_diagnoses(diagnoses: pd.DataFrame, hadm_ids: pd.Series) -> pd.Series:
    """
    Compute Charlson Comorbidity Index for each hadm_id.
    Uses ICD-10 codes only (icd_version == 10).
    """
    logger.info("Computing Charlson Comorbidity Index...")

    dx10 = diagnoses[diagnoses["icd_version"] == 10].copy()
    dx10["icd_clean"] = dx10["icd_code"].str.replace(".", "", regex=False).str.upper()

    # Build per-admission code set
    hadm_codes = dx10.groupby("hadm_id")["icd_clean"].apply(set).to_dict()

    scores = {}
    for hadm_id in hadm_ids:
        codes = hadm_codes.get(hadm_id, set())
        score = 0
        seen_conditions = set()
        for condition, info in CCI_ICD10_MAP.items():
            if condition in seen_conditions:
                continue
            for prefix in info["prefixes"]:
                if any(c.startswith(prefix) for c in codes):
                    score += info["points"]
                    seen_conditions.add(condition)
                    break
        scores[hadm_id] = score

    return pd.Series(scores)


def extract_diagnoses_features(diagnoses: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Add diagnosis-based features to the cohort."""
    logger.info("Extracting diagnosis features...")
    cohort = cohort.copy()

    # Get hadm_id from patient_id
    cohort["hadm_id"] = cohort["patient_id"].str.replace("MIMIC_", "").astype(int)

    # CCI
    cci_scores = compute_cci_from_diagnoses(diagnoses, cohort["hadm_id"])
    cohort["charlson_comorbidity_index"] = cohort["hadm_id"].map(cci_scores).fillna(0).astype(int)

    # Primary diagnosis (seq_num == 1)
    primary_dx = diagnoses[diagnoses["seq_num"] == 1][["hadm_id", "icd_code"]].copy()
    primary_dx.columns = ["hadm_id", "primary_diagnosis_code"]
    cohort = cohort.merge(primary_dx, on="hadm_id", how="left", suffixes=("_old", ""))
    if "primary_diagnosis_code_old" in cohort.columns:
        cohort.drop(columns=["primary_diagnosis_code_old"], inplace=True)
    cohort["primary_diagnosis_code"] = cohort["primary_diagnosis_code"].fillna("Z99.9")

    # Secondary diagnoses (seq_num 2-10)
    secondary = (
        diagnoses[(diagnoses["seq_num"] > 1) & (diagnoses["seq_num"] <= 10)]
        .groupby("hadm_id")["icd_code"]
        .apply(lambda x: "|".join(x.astype(str)))
        .reset_index()
    )
    secondary.columns = ["hadm_id", "secondary_diagnosis_codes"]
    cohort = cohort.merge(secondary, on="hadm_id", how="left", suffixes=("_old", ""))
    if "secondary_diagnosis_codes_old" in cohort.columns:
        cohort.drop(columns=["secondary_diagnosis_codes_old"], inplace=True)
    cohort["secondary_diagnosis_codes"] = cohort["secondary_diagnosis_codes"].fillna("")

    return cohort


def extract_medication_features(prescriptions: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Add medication features to the cohort."""
    logger.info("Extracting medication features...")
    cohort = cohort.copy()

    if "hadm_id" not in cohort.columns:
        cohort["hadm_id"] = cohort["patient_id"].str.replace("MIMIC_", "").astype(int)

    # Count active medications per admission
    med_counts = prescriptions.groupby("hadm_id")["drug"].nunique().reset_index()
    med_counts.columns = ["hadm_id", "num_active_medications"]
    cohort = cohort.merge(med_counts, on="hadm_id", how="left")
    cohort["num_active_medications"] = cohort["num_active_medications"].fillna(0).astype(int)

    # High-risk medication flag
    rx_lower = prescriptions.copy()
    rx_lower["drug_lower"] = rx_lower["drug"].str.lower().fillna("")
    high_risk_mask = rx_lower["drug_lower"].apply(
        lambda d: any(kw in d for kw in HIGH_RISK_MED_KEYWORDS)
    )
    high_risk_hadm = rx_lower[high_risk_mask]["hadm_id"].unique()
    cohort["high_risk_medication_flag"] = cohort["hadm_id"].isin(high_risk_hadm).astype(int)

    return cohort


def extract_utilization_features(admissions: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Compute prior utilization features (prior admits, ED visits, readmissions)."""
    logger.info("Extracting utilization features...")
    cohort = cohort.copy()

    if "hadm_id" not in cohort.columns:
        cohort["hadm_id"] = cohort["patient_id"].str.replace("MIMIC_", "").astype(int)

    # Sort admissions
    adm_sorted = admissions.sort_values(["subject_id", "admittime"]).copy()

    # For each index admission, count prior admissions in last 6 months
    # Merge cohort with all admissions on subject_id
    cohort_keys = cohort[["hadm_id", "subject_id", "admittime"]].copy() if "subject_id" in cohort.columns else \
        cohort[["hadm_id"]].merge(admissions[["hadm_id", "subject_id", "admittime"]], on="hadm_id", how="left")

    prior_stats = []
    for _, row in cohort_keys.iterrows():
        subj = row["subject_id"]
        index_time = row["admittime"]
        cutoff_6mo = index_time - pd.Timedelta(days=182)
        cutoff_1yr = index_time - pd.Timedelta(days=365)

        prior = adm_sorted[
            (adm_sorted["subject_id"] == subj) &
            (adm_sorted["admittime"] < index_time)
        ]

        prior_6mo = prior[prior["admittime"] >= cutoff_6mo]
        prior_1yr = prior[prior["admittime"] >= cutoff_1yr]

        # ED visits: admission type contains EMERGENCY or EU
        ed_mask = prior_6mo["admission_type"].str.upper().str.contains("EMERGENCY|EU", na=False)

        prior_stats.append({
            "hadm_id": row["hadm_id"],
            "prior_admissions_6mo": len(prior_6mo),
            "prior_ed_visits_6mo": int(ed_mask.sum()),
            "prior_readmissions_1yr": len(prior_1yr),
        })

    prior_df = pd.DataFrame(prior_stats)
    cohort = cohort.merge(prior_df, on="hadm_id", how="left", suffixes=("_old", ""))
    for col in ["prior_admissions_6mo", "prior_ed_visits_6mo", "prior_readmissions_1yr"]:
        old_col = col + "_old"
        if old_col in cohort.columns:
            cohort.drop(columns=[old_col], inplace=True)
        cohort[col] = cohort[col].fillna(0).astype(int)

    return cohort


def extract_lab_features(labevents: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract last-known lab values before discharge."""
    if labevents.empty:
        logger.warning("No lab data — skipping lab features")
        for lab_name in KEY_LAB_ITEMS.values():
            cohort[f"lab_{lab_name}"] = np.nan
        return cohort

    logger.info("Extracting lab features...")
    cohort = cohort.copy()

    if "hadm_id" not in cohort.columns:
        cohort["hadm_id"] = cohort["patient_id"].str.replace("MIMIC_", "").astype(int)

    # Filter to key lab items
    labs = labevents[labevents["itemid"].isin(KEY_LAB_ITEMS.keys())].copy()
    labs["lab_name"] = labs["itemid"].map(KEY_LAB_ITEMS)

    # Get last value per admission per lab
    labs_sorted = labs.sort_values("charttime")
    last_labs = labs_sorted.groupby(["hadm_id", "lab_name"])["valuenum"].last().unstack()
    last_labs.columns = [f"lab_{c}" for c in last_labs.columns]
    last_labs = last_labs.reset_index()

    cohort = cohort.merge(last_labs, on="hadm_id", how="left")
    return cohort
