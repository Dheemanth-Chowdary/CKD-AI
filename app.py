
import json
from pathlib import Path
import io

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"

bundle = joblib.load(ART / "ckd_model.joblib")
MODEL = bundle["model"]
FEATURES = bundle["features"]
THRESHOLDS = bundle["risk_thresholds"]

with open(ART / "metadata.json", "r", encoding="utf-8") as f:
    META = json.load(f)

with open(ART / "metrics.json", "r", encoding="utf-8") as f:
    METRICS = json.load(f)

with open(ART / "shap_importance.json", "r", encoding="utf-8") as f:
    SHAP_IMPORTANCE = json.load(f)

@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(MODEL)

EXPLAINER = get_explainer()

DISPLAY = {
    "age": "Age",
    "bp": "Blood Pressure",
    "sg": "Specific Gravity",
    "al": "Albumin",
    "su": "Sugar",
    "rbc": "Red Blood Cells",
    "pc": "Pus Cell",
    "pcc": "Pus Cell Clumps",
    "ba": "Bacteria",
    "bgr": "Blood Glucose Random",
    "bu": "Blood Urea",
    "sc": "Serum Creatinine",
    "sod": "Sodium",
    "pot": "Potassium",
    "hemo": "Haemoglobin",
    "pcv": "Packed Cell Volume",
    "wc": "White Blood Cell Count",
    "rc": "Red Blood Cell Count",
    "htn": "Hypertension",
    "dm": "Diabetes Mellitus",
    "cad": "Coronary Artery Disease",
    "appet": "Appetite",
    "pe": "Pedal Edema",
    "ane": "Anaemia",
}

HELP = {
    "rbc": "Dataset encoding: 0/1. Use the same encoding used during model training.",
    "pc": "Dataset encoding: 0/1.",
    "pcc": "Dataset encoding: 0/1.",
    "ba": "Dataset encoding: 0/1.",
    "htn": "Dataset encoding: 0/1.",
    "dm": "Dataset encoding: 0/1.",
    "cad": "Dataset encoding: 0/1.",
    "appet": "Dataset encoding: 0/1.",
    "pe": "Dataset encoding: 0/1.",
    "ane": "Dataset encoding: 0/1.",
}

st.set_page_config(
    page_title="CKD-AI | Explainable CKD Risk Assessment",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.metric-card {border: 1px solid #d9dee7; border-radius: 12px; padding: 16px; background: #fff;}
.warning-box {padding: 12px; border-radius: 10px; background: #fff4e5; border: 1px solid #f0c36d;}
.disclaimer {padding: 12px; border-radius: 10px; background: #f4f6f8; border: 1px solid #d7dce2; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("🩺 CKD-AI")
st.subheader("Explainable AI-Based Chronic Kidney Disease Risk Assessment")
st.caption("Research prototype based on the supplied CKD dataset and XGBoost + SHAP pipeline.")

st.markdown("""
<div class="disclaimer">
<b>Research / screening prototype:</b> This application is not a medical diagnostic device and does not
replace clinician assessment, eGFR/ACR evaluation, laboratory confirmation, or professional medical advice.
Risk thresholds are research-design thresholds, not clinically calibrated cut-offs.
</div>
""", unsafe_allow_html=True)

tab_assess, tab_explain, tab_perf, tab_about = st.tabs(
    ["🧪 Patient Assessment", "🔎 Explainability", "📊 Model Performance", "ℹ️ Research Details"]
)

def default_value(col):
    v = float(pd.read_csv(BASE / "ckd_dataset.csv")[col].median())
    return v

def get_default(col):
    rng = META["dataset_ranges"][col]
    if col in binary_features:
        return int(round(default_value(col)))
    if col in ["sg"]:
        return float(default_value(col))
    if col in ["al", "su"]:
        return int(round(default_value(col)))
    return float(default_value(col))

with tab_assess:
    st.markdown("### Enter the 24 clinical attributes")
    st.caption("The model expects the same numeric encoding used in the supplied dataset. No identifying information is required.")

    values = {}

    st.markdown("#### Basic measurements")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        values["age"] = st.number_input("Age", min_value=0.0, value=get_default("age"), step=1.0)
    with c2:
        values["bp"] = st.number_input("Blood Pressure", min_value=0.0, value=get_default("bp"), step=1.0)
    with c3:
        values["sg"] = st.number_input("Specific Gravity", min_value=0.0, value=get_default("sg"), step=0.001, format="%.3f")
    with c4:
        values["al"] = st.number_input("Albumin", min_value=0.0, value=get_default("al"), step=1.0)

    st.markdown("#### Urine-related attributes")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        values["su"] = st.number_input("Sugar", min_value=0.0, value=get_default("su"), step=1.0)
    with c2:
        values["rbc"] = st.selectbox("Red Blood Cells (0/1)", [0, 1], index=int(get_default("rbc")), help=HELP["rbc"])
    with c3:
        values["pc"] = st.selectbox("Pus Cell (0/1)", [0, 1], index=int(get_default("pc")), help=HELP["pc"])
    with c4:
        values["pcc"] = st.selectbox("Pus Cell Clumps (0/1)", [0, 1], index=int(get_default("pcc")), help=HELP["pcc"])
    with c5:
        values["ba"] = st.selectbox("Bacteria (0/1)", [0, 1], index=int(get_default("ba")), help=HELP["ba"])

    st.markdown("#### Blood-test attributes")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        values["bgr"] = st.number_input("Blood Glucose Random", min_value=0.0, value=get_default("bgr"), step=1.0)
    with c2:
        values["bu"] = st.number_input("Blood Urea", min_value=0.0, value=get_default("bu"), step=1.0)
    with c3:
        values["sc"] = st.number_input("Serum Creatinine", min_value=0.0, value=max(0.0, get_default("sc")), step=0.1)
    with c4:
        values["sod"] = st.number_input("Sodium", min_value=0.0, value=get_default("sod"), step=0.1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        values["pot"] = st.number_input("Potassium", min_value=0.0, value=max(0.0, get_default("pot")), step=0.1)
    with c2:
        values["hemo"] = st.number_input("Haemoglobin", min_value=0.0, value=get_default("hemo"), step=0.1)
    with c3:
        values["pcv"] = st.number_input("Packed Cell Volume", min_value=0.0, value=get_default("pcv"), step=0.1)
    with c4:
        values["wc"] = st.number_input("White Blood Cell Count", min_value=0.0, value=get_default("wc"), step=10.0)

    c1, c2 = st.columns(2)
    with c1:
        values["rc"] = st.number_input("Red Blood Cell Count", min_value=0.0, value=get_default("rc"), step=0.1)
    with c2:
        st.info("Use the same measurement units and encoding conventions as the research dataset.")

    st.markdown("#### Medical-condition flags")
    c1, c2, c3, c4, c5 = st.columns(5)
    for widget_col, col in zip([c1, c2, c3, c4, c5], ["htn", "dm", "cad", "appet", "pe"]):
        with widget_col:
            values[col] = st.selectbox(DISPLAY[col] + " (0/1)", [0, 1], index=int(get_default(col)), help=HELP[col])
    c1, c2 = st.columns(2)
    with c1:
        values["ane"] = st.selectbox("Anaemia (0/1)", [0, 1], index=int(get_default("ane")), help=HELP["ane"])

    st.divider()
    analyze = st.button("🔍 Analyze CKD Risk", type="primary", use_container_width=True)

    if analyze:
        row = pd.DataFrame([[values[c] for c in FEATURES]], columns=FEATURES)

        # Research-dataset range warning: do not silently clip or alter user data.
        outside = []
        for c in FEATURES:
            lo = META["dataset_ranges"][c]["min"]
            hi = META["dataset_ranges"][c]["max"]
            val = float(row.iloc[0][c])
            if val < lo or val > hi:
                outside.append(f"{DISPLAY[c]} ({val:g}; dataset range {lo:g}–{hi:g})")

        if outside:
            st.warning(
                "One or more values are outside the empirical range of the supplied training dataset. "
                "The app will still send the values to the model without clipping. "
                "This may reduce reliability. " + "; ".join(outside[:6])
            )

        p = float(MODEL.predict_proba(row)[:, 1][0])
        label = "CKD Positive" if p >= 0.50 else "Not CKD"
        if p < THRESHOLDS["moderate"]:
            tier = "Low"
        elif p < THRESHOLDS["high"]:
            tier = "Moderate"
        else:
            tier = "High"

        shap_values = EXPLAINER.shap_values(row)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = np.asarray(shap_values).reshape(-1)
        explanation = pd.DataFrame({
            "Feature": [DISPLAY.get(c, c) for c in FEATURES],
            "Feature_Code": FEATURES,
            "SHAP": shap_values,
            "Absolute_SHAP": np.abs(shap_values),
            "Value": [float(row.iloc[0][c]) for c in FEATURES],
        }).sort_values("Absolute_SHAP", ascending=False)

        st.session_state["last_result"] = {
            "row": row,
            "probability": p,
            "label": label,
            "tier": tier,
            "explanation": explanation,
        }

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        st.markdown("### Assessment Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", r["label"])
        m2.metric("CKD Probability", f"{r['probability']*100:.1f}%")
        m3.metric("Risk Tier", r["tier"])

        st.progress(min(max(r["probability"], 0.0), 1.0), text=f"Model probability: {r['probability']*100:.1f}%")

        st.markdown("#### Top contributing factors")
        top = r["explanation"].head(8).copy()
        top["Direction"] = np.where(top["SHAP"] >= 0, "Increases CKD model output", "Decreases CKD model output")
        top["Impact"] = top["SHAP"].abs().round(4)
        st.dataframe(top[["Feature", "Value", "SHAP", "Direction", "Impact"]], use_container_width=True, hide_index=True)

        result_json = {
            "prediction": r["label"],
            "ckd_probability": r["probability"],
            "risk_tier": r["tier"],
            "top_factors": top[["Feature", "Value", "SHAP", "Direction"]].to_dict(orient="records"),
            "model": "XGBoost",
            "research_note": "Prototype screening output; not a diagnosis."
        }
        st.download_button(
            "⬇️ Download Assessment JSON",
            data=json.dumps(result_json, indent=2),
            file_name="ckd_assessment_result.json",
            mime="application/json",
        )

with tab_explain:
    st.markdown("### Explainability")
    st.write("The deployed XGBoost model is explained using SHAP TreeExplainer, consistent with the research design.")
    imp = pd.DataFrame(SHAP_IMPORTANCE[:12])
    imp["Feature"] = imp["feature"].map(lambda x: DISPLAY.get(x, x))
    imp = imp.rename(columns={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature Code"})
    st.bar_chart(imp.set_index("Feature")["Mean |SHAP|"])
    st.dataframe(imp[["Feature", "Feature Code", "Mean |SHAP|"]], use_container_width=True, hide_index=True)

    if "last_result" in st.session_state:
        st.markdown("#### Latest patient-level explanation")
        e = st.session_state["last_result"]["explanation"].head(10).copy()
        e["Feature"] = e["Feature"]
        e = e.set_index("Feature")["SHAP"].sort_values()
        st.bar_chart(e)
    else:
        st.info("Run an assessment first to display the patient-level SHAP explanation.")

with tab_perf:
    st.markdown("### Model Performance")
    st.caption("These values are from the model artifact trained from the supplied dataset in this application build.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{METRICS['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{METRICS['precision']*100:.2f}%")
    c3.metric("Recall", f"{METRICS['recall']*100:.2f}%")
    c4.metric("AUC", f"{METRICS['auc']:.3f}")

    cm = np.array(METRICS["confusion_matrix"])
    st.markdown("#### XGBoost confusion matrix")
    cm_df = pd.DataFrame(cm, index=["Actual Not CKD", "Actual CKD"], columns=["Predicted Not CKD", "Predicted CKD"])
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("#### Dataset")
    d1, d2, d3 = st.columns(3)
    d1.metric("Records", META["dataset_shape"][0])
    d2.metric("Features", len(FEATURES))
    d3.metric("CKD / Not CKD", f"{META['class_distribution']['CKD']} / {META['class_distribution']['Not CKD']}")

with tab_about:
    st.markdown("### Research Implementation")
    st.markdown("""
    **Module 1 — Preprocessing & feature engineering**
    - 24 clinical attributes
    - Input validation
    - Dataset-consistent numeric/binary encoding

    **Module 2 — Prediction**
    - XGBoost deployment model
    - 250 trees
    - Maximum depth 4
    - Learning rate 0.08

    **Module 3 — Explainability & risk assessment**
    - SHAP TreeExplainer
    - CKD probability
    - Low / Moderate / High research risk tier
    - Ranked contributing features
    """)

    st.markdown("### Research dataset")
    st.write("400 records; 248 CKD and 152 non-CKD cases. The application does not collect or store names, national IDs, or other direct identifiers.")

    st.markdown("""
    <div class="warning-box">
    <b>Important:</b> The supplied report states that the working dataset is a single-context, 400-record
    research dataset and that the risk thresholds are not clinically calibrated. This app therefore demonstrates
    the research prototype and should not be used to diagnose or treat a patient.
    </div>
    """, unsafe_allow_html=True)
