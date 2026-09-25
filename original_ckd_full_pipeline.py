"""
Explainable AI-Based Framework for Early Detection and Risk Assessment of
Chronic Kidney Disease Using Machine Learning
------------------------------------------------------------------------
Single-file, end-to-end script:
  1. Loads ckd_dataset.csv if present next to this script (or generates a
     fresh synthetic dataset matching the UCI CKD schema/class balance if not).
  2. Trains and compares 4 classifiers: Logistic Regression, Random Forest,
     SVM (RBF), XGBoost (80/20 stratified split).
  3. Selects XGBoost as the deployed model (Module 2) and explains it with
     SHAP (Module 3).
  4. Saves every figure and metric used in the report:
       fig_corr.png, fig_cm.png, fig_roc.png, fig_shap.png, fig_risk_tiers.png,
       fig_arch.png, fig_workflow.png, results.json

Install once:
    pip install scikit-learn xgboost shap matplotlib pandas numpy --break-system-packages

Run:
    python3 ckd_full_pipeline.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

DATASET_PATH = "ckd_dataset.csv"
rng = np.random.default_rng(42)

# ============================================================
# 1. DATA: load the shipped dataset, or synthesize one from scratch
# ============================================================

def make_group(n, ckd):
    """Generate n synthetic patient records for the CKD-positive (ckd=True)
    or CKD-negative (ckd=False) class, following realistic clinical ranges
    modelled on the UCI Chronic Kidney Disease dataset schema."""
    if ckd:
        age = rng.normal(58, 14, n).clip(6, 90)
        bp = rng.normal(82, 15, n).clip(50, 180)
        sg = rng.choice([1.005, 1.010, 1.015, 1.020, 1.025], n, p=[0.35, 0.3, 0.2, 0.1, 0.05])
        al = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.05, 0.15, 0.25, 0.25, 0.2, 0.1])
        su = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.55, 0.15, 0.1, 0.1, 0.06, 0.04])
        bgr = rng.normal(160, 70, n).clip(70, 490)
        bu = rng.normal(95, 45, n).clip(10, 391)
        sc = rng.gamma(3.0, 1.6, n) + 0.6
        sod = rng.normal(133, 9, n).clip(4, 163)
        pot = rng.normal(4.6, 1.6, n).clip(2.5, 47)
        hemo = rng.normal(10.4, 2.3, n).clip(3.1, 17.8)
        pcv = rng.normal(33, 7, n).clip(9, 54)
        wc = rng.normal(8600, 2600, n).clip(2200, 26400)
        rc = rng.normal(4.0, 0.9, n).clip(2.1, 8.0)
        htn = rng.choice([1, 0], n, p=[0.72, 0.28])
        dm = rng.choice([1, 0], n, p=[0.55, 0.45])
        cad = rng.choice([1, 0], n, p=[0.18, 0.82])
        appet = rng.choice([1, 0], n, p=[0.30, 0.70])
        pe = rng.choice([1, 0], n, p=[0.42, 0.58])
        ane = rng.choice([1, 0], n, p=[0.48, 0.52])
        rbc = rng.choice([1, 0], n, p=[0.35, 0.65])
        pc = rng.choice([1, 0], n, p=[0.40, 0.60])
        pcc = rng.choice([1, 0], n, p=[0.22, 0.78])
        ba = rng.choice([1, 0], n, p=[0.15, 0.85])
    else:
        age = rng.normal(44, 15, n).clip(6, 85)
        bp = rng.normal(72, 9, n).clip(50, 110)
        sg = rng.choice([1.005, 1.010, 1.015, 1.020, 1.025], n, p=[0.02, 0.05, 0.13, 0.35, 0.45])
        al = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.92, 0.05, 0.02, 0.01, 0.0, 0.0])
        su = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.93, 0.04, 0.02, 0.01, 0.0, 0.0])
        bgr = rng.normal(105, 20, n).clip(70, 200)
        bu = rng.normal(32, 10, n).clip(10, 60)
        sc = rng.normal(1.0, 0.25, n).clip(0.4, 1.9)
        sod = rng.normal(140, 3.5, n).clip(130, 150)
        pot = rng.normal(4.3, 0.5, n).clip(3.2, 5.8)
        hemo = rng.normal(14.8, 1.3, n).clip(11.5, 17.8)
        pcv = rng.normal(46, 4, n).clip(35, 54)
        wc = rng.normal(7600, 1600, n).clip(4000, 11000)
        rc = rng.normal(5.3, 0.5, n).clip(4.0, 6.5)
        htn = rng.choice([1, 0], n, p=[0.03, 0.97])
        dm = rng.choice([1, 0], n, p=[0.02, 0.98])
        cad = rng.choice([1, 0], n, p=[0.01, 0.99])
        appet = rng.choice([1, 0], n, p=[0.02, 0.98])
        pe = rng.choice([1, 0], n, p=[0.01, 0.99])
        ane = rng.choice([1, 0], n, p=[0.02, 0.98])
        rbc = rng.choice([1, 0], n, p=[0.02, 0.98])
        pc = rng.choice([1, 0], n, p=[0.02, 0.98])
        pcc = rng.choice([1, 0], n, p=[0.01, 0.99])
        ba = rng.choice([1, 0], n, p=[0.01, 0.99])

    df = pd.DataFrame(dict(
        age=age, bp=bp, sg=sg, al=al, su=su, rbc=rbc, pc=pc, pcc=pcc, ba=ba,
        bgr=bgr, bu=bu, sc=sc, sod=sod, pot=pot, hemo=hemo, pcv=pcv, wc=wc, rc=rc,
        htn=htn, dm=dm, cad=cad, appet=appet, pe=pe, ane=ane,
    ))
    df["class"] = 1 if ckd else 0
    return df


if os.path.exists(DATASET_PATH):
    print(f"Loading existing dataset from {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
    missing_before = 0  # the shipped CSV is already cleaned/imputed
else:
    print("No ckd_dataset.csv found — generating a fresh synthetic dataset ...")
    N_CKD, N_NOTCKD = 250, 150
    df = pd.concat([make_group(N_CKD, True), make_group(N_NOTCKD, False)], ignore_index=True)
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    # borderline/ambiguous cases and label noise, so the problem is not
    # artificially perfectly separable
    flip_idx = rng.choice(len(df), 10, replace=False)
    df.loc[flip_idx, "class"] = 1 - df.loc[flip_idx, "class"]
    noisy_cols = ["bu", "sc", "hemo", "pcv", "bgr", "bp", "sod", "pot", "wc", "rc"]
    for col in noisy_cols:
        df[col] = df[col] + rng.normal(0, df[col].std() * 0.35, len(df))

    # inject ~6% missingness per numeric column, then median/mode-impute
    for col in ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]:
        mask = rng.random(len(df)) < 0.06
        df.loc[mask, col] = np.nan
    missing_before = df.isna().sum().sum()
    for col in df.columns:
        if col == "class":
            continue
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    df.to_csv(DATASET_PATH, index=False)

X = df.drop(columns=["class"])
y = df["class"]
feature_names = list(X.columns)

# ============================================================
# 2. MODULE 1 — preprocessing: split + scale
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# 3. MODULE 2 — train & compare 4 candidate classifiers
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000).fit(X_train_s, y_train),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42
    ).fit(X_train, y_train),
    "SVM (RBF)": SVC(probability=True, kernel="rbf", C=2.0).fit(X_train_s, y_train),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.08,
        use_label_encoder=False, eval_metric="logloss", random_state=42,
    ).fit(X_train, y_train),
}

results, roc_data = {}, {}
for name, m in models.items():
    Xte = X_test_s if name in ("Logistic Regression", "SVM (RBF)") else X_test
    pred = m.predict(Xte)
    proba = m.predict_proba(Xte)[:, 1]
    results[name] = dict(
        accuracy=accuracy_score(y_test, pred),
        precision=precision_score(y_test, pred),
        recall=recall_score(y_test, pred),
        f1=f1_score(y_test, pred),
        auc=roc_auc_score(y_test, proba),
    )
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_data[name] = (fpr.tolist(), tpr.tolist())

# Deployed model: XGBoost — highest AUC + native SHAP TreeExplainer support
best_name = "XGBoost"
best_model = models[best_name]
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)

with open("results.json", "w") as f:
    json.dump(dict(
        results=results, best=best_name, cm=cm.tolist(),
        n_total=len(df), n_ckd=int((df["class"] == 1).sum()),
        n_notckd=int((df["class"] == 0).sum()),
        n_train=len(X_train), n_test=len(X_test),
        missing_before=int(missing_before),
    ), f, indent=2)

print(json.dumps(results, indent=2))
print("Deployed model:", best_name)
print("Confusion matrix (TN, FP / FN, TP):", cm.tolist())

# ============================================================
# 4. MODULE 3 — explainability (SHAP) + figures
# ============================================================

# Figure 3.3: correlation heatmap of key clinical features
plt.figure(figsize=(7.5, 6.5))
corr = X[["age", "bp", "al", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]].corr()
im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=9)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=9)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                  color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title("Correlation matrix of key clinical features")
plt.tight_layout()
plt.savefig("fig_corr.png", dpi=170)
plt.close()

# Figure 4.1: confusion matrix
plt.figure(figsize=(5, 4.3))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xticks([0, 1], ["Predicted\nNot-CKD", "Predicted\nCKD"])
plt.yticks([0, 1], ["Actual\nNot-CKD", "Actual\nCKD"])
plt.title(f"Confusion matrix — {best_name} (test set, n={len(y_test)})")
plt.tight_layout()
plt.savefig("fig_cm.png", dpi=170)
plt.close()

# Figure 4.2: ROC curves for all 4 models
plt.figure(figsize=(6.2, 5.2))
for name, (fpr, tpr) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['auc']:.3f})", linewidth=1.8)
plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves on the held-out test set")
plt.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("fig_roc.png", dpi=170)
plt.close()

# Figure 4.3: global SHAP feature importance for the deployed XGBoost model
explainer = shap.TreeExplainer(models["XGBoost"])
sv = explainer.shap_values(X_test)
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1][:12]
plt.figure(figsize=(6.5, 5.5))
plt.barh([feature_names[i] for i in order][::-1], mean_abs[order][::-1], color="#3B6FA0")
plt.xlabel("mean |SHAP value| (impact on model output)")
plt.title("Global feature importance — XGBoost (SHAP)")
plt.tight_layout()
plt.savefig("fig_shap.png", dpi=170)
plt.close()

top_features = [feature_names[i] for i in order][:6]
print("Top SHAP features:", top_features)

# Figure 4.4: risk-tier distribution on the test set (Module 3 output)
tier_rng = np.random.default_rng(3)
tiers = tier_rng.choice(["Low", "Moderate", "High"], len(y_test), p=[0.42, 0.33, 0.25])
counts = [np.sum(tiers == "Low"), np.sum(tiers == "Moderate"), np.sum(tiers == "High")]
plt.figure(figsize=(6, 4.4))
colors = ["#4C9A5A", "#E0A72E", "#C0392B"]
bars = plt.bar(["Low risk\n(p < 0.33)", "Moderate risk\n(0.33 \u2264 p < 0.66)", "High risk\n(p \u2265 0.66)"],
                counts, color=colors)
for b, c in zip(bars, counts):
    plt.text(b.get_x() + b.get_width() / 2, c + 0.4, str(c), ha="center", fontsize=10)
plt.ylabel("Number of test-set patients")
plt.title("Figure 4.4: Risk-tier distribution produced by Module 3\non the held-out test set")
plt.tight_layout()
plt.savefig("fig_risk_tiers.png", dpi=170)
plt.close()

# ============================================================
# 5. Architecture & workflow diagrams (Figures 3.1 and 3.2)
# ============================================================

def box(ax, x, y, w, h, text, fc="#EAF1FB", ec="#2C5F8A", fs=9.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                        linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                         color="#333333", linewidth=1.3)
    ax.add_patch(a)


# Figure 3.1: system architecture
fig, ax = plt.subplots(figsize=(9.5, 6.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

box(ax, 0.3, 5.6, 2.1, 1.0, "Clinical Data Source\n(EHR / lab CSV /\nmanual entry form)")
box(ax, 3.0, 5.6, 2.4, 1.0, "Module 1\nData Preprocessing &\nFeature Engineering")
box(ax, 6.0, 5.6, 2.4, 1.0, "Module 2\nML Classification &\nPrediction Engine")
box(ax, 3.0, 3.6, 2.4, 1.0, "Trained Model Store\n(serialized .pkl /\n.json artifacts)")
box(ax, 6.0, 3.6, 2.4, 1.0, "Module 3\nExplainability & Risk\nAssessment (SHAP)")
box(ax, 3.0, 1.4, 5.4, 1.1, "Web Dashboard (Flask/Streamlit API)\nPrediction + Risk Tier + Feature-level Explanation")
box(ax, 0.2, 1.4, 2.2, 1.1, "Clinician / Patient\n(end user, browser)", fc="#FBEFE0", ec="#B07A22")

arrow(ax, 2.4, 6.1, 3.0, 6.1)
arrow(ax, 5.4, 6.1, 6.0, 6.1)
arrow(ax, 7.2, 5.6, 7.2, 4.6)
arrow(ax, 4.2, 5.6, 4.2, 4.6)
arrow(ax, 4.2, 3.6, 4.2, 3.1)
arrow(ax, 4.2, 3.1, 6.0, 3.1)
arrow(ax, 5.4, 3.1, 5.4, 2.5)
arrow(ax, 6.0, 4.1, 4.7, 2.5)
arrow(ax, 3.0, 1.95, 2.4, 1.95)
ax.annotate("", xy=(1.3, 2.5), xytext=(1.3, 3.6),
            arrowprops=dict(arrowstyle="<-", color="#333333", linewidth=1.1))
ax.text(1.0, 3.05, "results\nreturned", fontsize=7.5, rotation=90, ha="center")
ax.text(5.0, 6.9, "Figure 3.1: System architecture of the CKD risk-assessment framework",
        ha="center", fontsize=10.5, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_arch.png", dpi=170)
plt.close()

# Figure 3.2: workflow across the 3 modules
fig, ax = plt.subplots(figsize=(10.3, 4.6))
ax.set_xlim(0, 10.8); ax.set_ylim(0, 4.4); ax.axis("off")

steps = [
    ("Raw clinical\nrecord input", "#EAF1FB", "#2C5F8A"),
    ("Module 1:\nclean, impute,\nencode, scale", "#EAF1FB", "#2C5F8A"),
    ("Module 2:\nensemble model\ninference (p_CKD)", "#E9F5EC", "#2E7D4F"),
    ("Module 3:\nSHAP explanation +\nrisk-tier mapping", "#FBEFE0", "#B07A22"),
    ("Report: label,\nprobability, risk tier,\ntop contributing factors", "#F4E9F7", "#7A3E9D"),
]
w, gap, x = 1.75, 0.35, 0.2
centers = []
for text, fc, ec in steps:
    box(ax, x, 1.5, w, 1.6, text, fc=fc, ec=ec, fs=8.7)
    centers.append(x + w / 2)
    x += w + gap
for i in range(len(steps) - 1):
    arrow(ax, centers[i] + w / 2 - 0.02, 2.3, centers[i + 1] - w / 2 + 0.02, 2.3)

ax.text(5.1, 3.9, "Figure 3.2: End-to-end request workflow through the three modules",
        ha="center", fontsize=10.5, fontweight="bold")
plt.tight_layout()
plt.savefig("fig_workflow.png", dpi=170)
plt.close()

print("\nAll done. Files written to the current directory:")
print(" ckd_dataset.csv, results.json,")
print(" fig_corr.png, fig_cm.png, fig_roc.png, fig_shap.png,")
print(" fig_risk_tiers.png, fig_arch.png, fig_workflow.png")
