
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"

bundle = joblib.load(ART / "ckd_model.joblib")
MODEL = bundle["model"]
FEATURES = bundle["features"]
THRESHOLDS = bundle["risk_thresholds"]

with open(ART / "metadata.json", "r", encoding="utf-8") as f:
    META = json.load(f)

EXPLAINER = shap.TreeExplainer(MODEL)

app = FastAPI(
    title="CKD-AI Explainable Risk Assessment API",
    version="1.0.0",
    description="Research prototype API for CKD screening-risk prediction using XGBoost and SHAP."
)

class PatientRecord(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    su: float
    rbc: int = Field(ge=0, le=1)
    pc: int = Field(ge=0, le=1)
    pcc: int = Field(ge=0, le=1)
    ba: int = Field(ge=0, le=1)
    bgr: float
    bu: float
    sc: float
    sod: float
    pot: float
    hemo: float
    pcv: float
    wc: float
    rc: float
    htn: int = Field(ge=0, le=1)
    dm: int = Field(ge=0, le=1)
    cad: int = Field(ge=0, le=1)
    appet: int = Field(ge=0, le=1)
    pe: int = Field(ge=0, le=1)
    ane: int = Field(ge=0, le=1)

@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "features": len(FEATURES)}

@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost",
        "risk_thresholds": THRESHOLDS,
        "dataset_shape": META["dataset_shape"],
        "class_distribution": META["class_distribution"],
    }

@app.post("/predict")
def predict(record: PatientRecord):
    row = pd.DataFrame([[getattr(record, f) for f in FEATURES]], columns=FEATURES)
    p = float(MODEL.predict_proba(row)[:, 1][0])
    label = "CKD Positive" if p >= 0.50 else "Not CKD"
    if p < THRESHOLDS["moderate"]:
        tier = "Low"
    elif p < THRESHOLDS["high"]:
        tier = "Moderate"
    else:
        tier = "High"

    sv = EXPLAINER.shap_values(row)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv).reshape(-1)
    order = np.argsort(np.abs(sv))[::-1][:8]

    factors = [
        {
            "feature": FEATURES[i],
            "shap": float(sv[i]),
            "direction": "increases CKD model output" if sv[i] >= 0 else "decreases CKD model output",
            "value": float(row.iloc[0, i]),
        }
        for i in order
    ]

    return {
        "prediction": label,
        "ckd_probability": p,
        "risk_tier": tier,
        "top_contributing_factors": factors,
        "model": "XGBoost",
        "disclaimer": "Research screening prototype; not a diagnosis."
    }
