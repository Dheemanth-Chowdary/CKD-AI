
# CKD-AI — Explainable CKD Risk Assessment App

A Research Methodology application based on the supplied project:

**EXPLAINABLE AI-BASED FRAMEWORK FOR EARLY DETECTION AND RISK ASSESSMENT OF CHRONIC KIDNEY DISEASE USING MACHINE LEARNING**

## What is included

- `app.py` — Streamlit web application
- `api.py` — FastAPI REST API
- `train_model.py` — explicit retraining script
- `ckd_dataset.csv` — supplied research dataset
- `artifacts/ckd_model.joblib` — trained XGBoost deployment artifact
- `artifacts/metrics.json` — test-set metrics for this build
- `artifacts/shap_importance.json` — global SHAP importance
- `artifacts/metadata.json` — feature and dataset metadata
- `requirements.txt` — Python dependencies

## Run the web application

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Run the REST API

```bash
uvicorn api:app --reload
```

API documentation will be available at:

`http://127.0.0.1:8000/docs`

## Retrain

Only retrain intentionally from the supplied dataset:

```bash
python train_model.py
```

The retraining script does not silently generate synthetic data.

## Research workflow

Patient input
→ validation/range warning
→ XGBoost
→ CKD probability
→ Low/Moderate/High research risk tier
→ SHAP feature attribution
→ explanation

## Important research limitation

This is a research prototype / screening-support application. It is not a diagnostic device.
The supplied report states that the working dataset is a single-context 400-record dataset and
that the 0.33 / 0.66 risk thresholds are fixed design thresholds rather than clinically calibrated cut-offs.
Clinical deployment would require appropriate validation, governance and regulatory review.
