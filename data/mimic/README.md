# MIMIC-IV Data Directory

Place your MIMIC-IV files here before running the pipeline.

## Required folder structure

```
data/mimic/
└── hosp/
    ├── admissions.csv       ← Required
    ├── patients.csv         ← Required
    ├── diagnoses_icd.csv    ← Required
    ├── prescriptions.csv    ← Required
    ├── labevents.csv        ← Optional (large ~7GB, adds lab features)
    ├── d_labitems.csv       ← Optional (lab item dictionary)
    └── procedures_icd.csv   ← Optional
```

## How to get MIMIC-IV

1. Create a PhysioNet account: https://physionet.org/register/
2. Complete CITI training (takes ~2 hours, free)
3. Apply for MIMIC-IV access: https://physionet.org/content/mimiciv/
4. Download the `hosp` module files once approved (usually 2-3 days)

## Running the pipeline

```bash
# Full pipeline with all patients
python -m src.data.mimic.pipeline

# Test run with 1000 patients (fast)
python -m src.data.mimic.pipeline --max-patients 1000

# Skip large labevents file
python -m src.data.mimic.pipeline --lab-nrows 0
```

After the pipeline completes, run the model trainer as usual:
```bash
python -m src.models.model_trainer
```
