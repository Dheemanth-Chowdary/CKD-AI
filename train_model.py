
"""Train the deployment artifact from the supplied ckd_dataset.csv.

This script intentionally does NOT generate synthetic fallback data.
The application should only be retrained from an explicitly supplied dataset.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap

BASE = Path(__file__).resolve().parent
DATA = BASE / "ckd_dataset.csv"
ART = BASE / "artifacts"
ART.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"Missing {DATA}. Place the supplied dataset in the project root.")

df = pd.read_csv(DATA)
features = [c for c in df.columns if c != "class"]
X = df[features]
y = df["class"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.08,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, pred)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred)),
    "recall": float(recall_score(y_test, pred)),
    "f1": float(f1_score(y_test, pred)),
    "auc": float(roc_auc_score(y_test, proba)),
    "confusion_matrix": cm.tolist(),
}

joblib.dump(
    {"model": model, "features": features, "risk_thresholds": {"moderate": 0.33, "high": 0.66}},
    ART / "ckd_model.joblib",
)

explainer = shap.TreeExplainer(model)
sv = np.asarray(explainer.shap_values(X_test))
if sv.ndim == 3:
    sv = sv[:, :, 1]
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1]
importance = [
    {"feature": features[int(i)], "mean_abs_shap": float(mean_abs[int(i)])}
    for i in order
]
(ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
(ART / "shap_importance.json").write_text(json.dumps(importance, indent=2))
print(json.dumps(metrics, indent=2))
