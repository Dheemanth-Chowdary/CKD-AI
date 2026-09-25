# CKD-AI UI v2.0 — verified Streamlit build

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="CKD-AI | Explainable Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "ckd_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
SHAP_PATH = ARTIFACT_DIR / "shap_importance.json"
DATASET_PATH = BASE_DIR / "ckd_dataset.csv"


FEATURES = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane",
]

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

BINARY_FEATURES = {
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"
}

ORDINAL_FEATURES = {"al", "su"}


# ============================================================
# PREMIUM UI
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .stApp {
        background: #f6f8fb;
    }

    [data-testid="stHeader"] {
        background: rgba(246,248,251,0.92);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1736 0%, #102554 100%);
    }

    [data-testid="stSidebar"] * {
        color: #eef4ff;
    }

    .brand {
        padding: 8px 4px 22px 4px;
        border-bottom: 1px solid rgba(255,255,255,.13);
        margin-bottom: 20px;
    }

    .brand-title {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .brand-subtitle {
        color: #a9bad8 !important;
        font-size: 12px;
        margin-top: 3px;
    }

    .side-card {
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 14px;
        padding: 14px;
        margin: 10px 0;
    }

    .side-label {
        color: #9eb1d4 !important;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .side-value {
        font-size: 18px;
        font-weight: 700;
        margin-top: 4px;
    }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(135deg, #0b1736 0%, #173a79 58%, #146c94 100%);
        border-radius: 24px;
        padding: 30px 34px;
        color: white;
        box-shadow: 0 12px 35px rgba(16,38,80,.16);
        margin-bottom: 22px;
    }

    .hero-kicker {
        color: #9fe7ff;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: 38px;
        line-height: 1.08;
        font-weight: 800;
        margin: 8px 0 10px;
        letter-spacing: -1px;
    }

    .hero-text {
        color: #d8e7fb;
        font-size: 15px;
        max-width: 850px;
        line-height: 1.6;
    }

    .hero-pill {
        display: inline-block;
        margin-top: 16px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.18);
        font-size: 12px;
    }

    /* ---------- Section ---------- */
    .section-title {
        font-size: 21px;
        font-weight: 800;
        color: #12203b;
        margin: 8px 0 4px;
    }

    .section-subtitle {
        color: #66738b;
        font-size: 13px;
        margin-bottom: 14px;
    }

    /* ---------- Cards ---------- */
    .info-card {
        background: white;
        border: 1px solid #e3e8f0;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 5px 18px rgba(28,45,78,.05);
    }

    .card-label {
        color: #738097;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .07em;
    }

    .card-value {
        color: #12203b;
        font-size: 25px;
        font-weight: 800;
        margin-top: 5px;
    }

    .card-caption {
        color: #7c879b;
        font-size: 12px;
        margin-top: 3px;
    }

    /* ---------- Status ---------- */
    .status-low {
        background: #eaf8f0;
        color: #14733f;
        border: 1px solid #bfe8cf;
        border-radius: 12px;
        padding: 13px 16px;
        font-weight: 700;
    }

    .status-moderate {
        background: #fff7e7;
        color: #9a6300;
        border: 1px solid #f0d28c;
        border-radius: 12px;
        padding: 13px 16px;
        font-weight: 700;
    }

    .status-high {
        background: #fff0f0;
        color: #b42318;
        border: 1px solid #f0b8b8;
        border-radius: 12px;
        padding: 13px 16px;
        font-weight: 700;
    }

    /* ---------- Notice ---------- */
    .notice {
        background: #ffffff;
        border: 1px solid #e0e7f0;
        border-left: 4px solid #1b73e8;
        border-radius: 12px;
        padding: 13px 16px;
        color: #4c5b72;
        font-size: 13px;
        line-height: 1.55;
    }

    .medical-note {
        background: #fff8ec;
        border: 1px solid #f1d59d;
        border-radius: 12px;
        padding: 13px 16px;
        color: #6f531c;
        font-size: 12px;
        line-height: 1.5;
    }

    /* ---------- Streamlit widgets ---------- */
    div[data-testid="stButton"] > button {
        border-radius: 11px;
        font-weight: 700;
        min-height: 46px;
    }

    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 11px;
        font-weight: 700;
        min-height: 46px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e3e8f0;
        border-radius: 14px;
        padding: 13px;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #8792a6;
        font-size: 11px;
        padding-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD FILES
# ============================================================

required = {
    "Model": MODEL_PATH,
    "Metadata": METADATA_PATH,
    "Metrics": METRICS_PATH,
    "SHAP importance": SHAP_PATH,
    "Dataset": DATASET_PATH,
}

missing = [
    f"{name}: {path}"
    for name, path in required.items()
    if not path.exists()
]

if missing:
    st.error("Required project files are missing.")
    for item in missing:
        st.write(f"- `{item}`")
    st.stop()


try:
    bundle = joblib.load(MODEL_PATH)
    MODEL = bundle["model"]
    FEATURES = list(bundle.get("features", FEATURES))
    RISK_THRESHOLDS = bundle.get(
        "risk_thresholds",
        {"moderate": 0.33, "high": 0.66},
    )
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


METADATA = load_json(METADATA_PATH)
METRICS = load_json(METRICS_PATH)
SHAP_IMPORTANCE = load_json(SHAP_PATH)

try:
    DATASET = pd.read_csv(DATASET_PATH)
except Exception as e:
    st.error("Unable to read ckd_dataset.csv.")
    st.exception(e)
    st.stop()


try:
    EXPLAINER = shap.TreeExplainer(MODEL)
except Exception as e:
    st.error("Unable to initialize the SHAP explainer.")
    st.exception(e)
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Compatibility with earlier app versions.
old = st.session_state.get("last_result")

if old is not None:
    if "prediction" not in old and "label" in old:
        old["prediction"] = old["label"]

    if "risk_tier" not in old and "tier" in old:
        old["risk_tier"] = old["tier"]

    required_result_keys = {
        "input",
        "probability",
        "prediction",
        "risk_tier",
        "explanation",
    }

    if not required_result_keys.issubset(old.keys()):
        st.session_state.last_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-title">🩺 CKD-AI</div>
            <div class="brand-subtitle">
                Explainable Risk Assessment Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-card">
            <div class="side-label">Model</div>
            <div class="side-value">XGBoost</div>
        </div>

        <div class="side-card">
            <div class="side-label">Explainability</div>
            <div class="side-value">SHAP</div>
        </div>

        <div class="side-card">
            <div class="side-label">Clinical Inputs</div>
            <div class="side-value">24 Features</div>
        </div>

        <div class="side-card">
            <div class="side-label">Purpose</div>
            <div class="side-value">Research / Screening</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="medical-note">
        <b>Important</b><br>
        This application is a research prototype.
        It does not provide a clinical diagnosis.
        Risk thresholds are not clinically calibrated.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Explainable Machine Learning</div>
        <div class="hero-title">
            Chronic Kidney Disease<br>
            Risk Assessment
        </div>
        <div class="hero-text">
            A research-oriented clinical screening interface that combines
            XGBoost prediction with SHAP-based explanations to show both
            the estimated CKD probability and the features contributing
            to the model output.
        </div>
        <div class="hero-pill">
            ● XGBoost &nbsp; • &nbsp; SHAP &nbsp; • &nbsp; 24 Clinical Features
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP SUMMARY
# ============================================================

result = st.session_state.get("last_result")

if result is None:

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-label">Prediction Model</div>
                <div class="card-value">XGBoost</div>
                <div class="card-caption">Trained research model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-label">Explainability</div>
                <div class="card-value">SHAP</div>
                <div class="card-caption">Feature-level attribution</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-label">Clinical Inputs</div>
                <div class="card-value">24</div>
                <div class="card-caption">Research dataset features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-label">Risk Levels</div>
                <div class="card-value">3</div>
                <div class="card-caption">Low • Moderate • High</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# NAVIGATION
# ============================================================

tab_assess, tab_explain, tab_performance, tab_research = st.tabs(
    [
        "🧪  Patient Assessment",
        "🔎  Explainability",
        "📊  Model Performance",
        "📚  Research Details",
    ]
)


# ============================================================
# PATIENT ASSESSMENT
# ============================================================

with tab_assess:

    st.markdown(
        '<div class="section-title">Patient Clinical Assessment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Enter all 24 clinical attributes. Fields are intentionally '
        'blank to prevent predictions from being generated from '
        'automatically inserted values.'
        '</div>',
        unsafe_allow_html=True,
    )

    # -------------------------------
    # BASIC
    # -------------------------------

    st.markdown("#### 01 · Basic Measurements")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=120.0,
            value=None,
            step=1.0,
            placeholder="Enter age",
        )

    with c2:
        bp = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=300.0,
            value=None,
            step=1.0,
            placeholder="Enter BP",
        )

    with c3:
        sg = st.number_input(
            "Specific Gravity",
            min_value=0.0,
            max_value=2.0,
            value=None,
            step=0.001,
            format="%.3f",
            placeholder="e.g. 1.020",
        )

    # -------------------------------
    # URINE
    # -------------------------------

    st.markdown("#### 02 · Urine Parameters")

    c1, c2, c3 = st.columns(3)

    with c1:
        al = st.selectbox(
            "Albumin",
            [0, 1, 2, 3, 4, 5],
            index=None,
            placeholder="Select value",
        )

    with c2:
        su = st.selectbox(
            "Sugar",
            [0, 1, 2, 3, 4, 5],
            index=None,
            placeholder="Select value",
        )

    with c3:
        rbc = st.selectbox(
            "Red Blood Cells",
            [0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        pc = st.selectbox(
            "Pus Cell",
            [0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    with c2:
        pcc = st.selectbox(
            "Pus Cell Clumps",
            [0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    with c3:
        ba = st.selectbox(
            "Bacteria",
            [0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    # -------------------------------
    # BLOOD
    # -------------------------------

    st.markdown("#### 03 · Blood Parameters")

    c1, c2, c3 = st.columns(3)

    with c1:
        bgr = st.number_input(
            "Blood Glucose Random",
            min_value=0.0,
            max_value=1000.0,
            value=None,
            step=1.0,
            placeholder="Enter value",
        )

    with c2:
        bu = st.number_input(
            "Blood Urea",
            min_value=0.0,
            max_value=500.0,
            value=None,
            step=1.0,
            placeholder="Enter value",
        )

    with c3:
        sc = st.number_input(
            "Serum Creatinine",
            min_value=0.0,
            max_value=100.0,
            value=None,
            step=0.1,
            placeholder="Enter value",
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        sod = st.number_input(
            "Sodium",
            min_value=0.0,
            max_value=250.0,
            value=None,
            step=0.1,
            placeholder="Enter value",
        )

    with c2:
        pot = st.number_input(
            "Potassium",
            min_value=0.0,
            max_value=20.0,
            value=None,
            step=0.1,
            placeholder="Enter value",
        )

    with c3:
        hemo = st.number_input(
            "Haemoglobin",
            min_value=0.0,
            max_value=30.0,
            value=None,
            step=0.1,
            placeholder="Enter value",
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        pcv = st.number_input(
            "Packed Cell Volume",
            min_value=0.0,
            max_value=100.0,
            value=None,
            step=0.1,
            placeholder="Enter value",
        )

    with c2:
        wc = st.number_input(
            "White Blood Cell Count",
            min_value=0.0,
            max_value=50000.0,
            value=None,
            step=10.0,
            placeholder="Enter value",
        )

    with c3:
        rc = st.number_input(
            "Red Blood Cell Count",
            min_value=0.0,
            max_value=20.0,
            value=None,
            step=0.1,
            placeholder="Enter value",
        )

    # -------------------------------
    # CONDITIONS
    # -------------------------------

    st.markdown("#### 04 · Medical Condition Indicators")

    c1, c2, c3 = st.columns(3)

    with c1:
        htn = st.selectbox(
            "Hypertension",
            [0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    with c2:
        dm = st.selectbox(
            "Diabetes Mellitus",
            [0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    with c3:
        cad = st.selectbox(
            "Coronary Artery Disease",
            [0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        appet = st.selectbox(
            "Appetite",
            [0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    with c2:
        pe = st.selectbox(
            "Pedal Edema",
            [0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    with c3:
        ane = st.selectbox(
            "Anaemia",
            [0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    values = {
        "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
        "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba,
        "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
        "hemo": hemo, "pcv": pcv, "wc": wc, "rc": rc,
        "htn": htn, "dm": dm, "cad": cad, "appet": appet,
        "pe": pe, "ane": ane,
    }

    missing_fields = [
        DISPLAY.get(feature, feature)
        for feature in FEATURES
        if values.get(feature) is None
    ]

    st.markdown("")

    if missing_fields:
        st.markdown(
            f'<div class="notice"><b>{len(missing_fields)} fields remaining</b> · '
            'Complete all clinical inputs to activate the assessment.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    analyze = st.button(
        "🔍  Analyze CKD Risk",
        type="primary",
        use_container_width=True,
        disabled=bool(missing_fields),
    )

    if analyze:

        try:

            model_values = []

            for feature in FEATURES:

                value = values[feature]

                if feature in BINARY_FEATURES or feature in ORDINAL_FEATURES:
                    value = int(value)
                else:
                    value = float(value)

                model_values.append(value)

            input_df = pd.DataFrame(
                [model_values],
                columns=FEATURES,
            )

            # Dataset range warnings
            dataset_ranges = METADATA.get(
                "dataset_ranges",
                {},
            )

            outside = []

            for feature in FEATURES:

                if feature not in dataset_ranges:
                    continue

                try:
                    lo = float(dataset_ranges[feature]["min"])
                    hi = float(dataset_ranges[feature]["max"])
                    current = float(input_df.iloc[0][feature])

                    if current < lo or current > hi:
                        outside.append(
                            f"{DISPLAY.get(feature, feature)} = {current:g} "
                            f"(dataset range {lo:g}–{hi:g})"
                        )
                except Exception:
                    pass

            if outside:
                st.warning(
                    "Some values are outside the empirical range of the "
                    "supplied training dataset. Values are passed to the "
                    "model without clipping."
                )

            # Prediction
            probability = float(
                MODEL.predict_proba(input_df)[0][1]
            )

            prediction = (
                "CKD Positive"
                if probability >= 0.50
                else "Not CKD"
            )

            moderate = float(
                RISK_THRESHOLDS.get("moderate", 0.33)
            )
            high = float(
                RISK_THRESHOLDS.get("high", 0.66)
            )

            if probability < moderate:
                risk_tier = "Low"
            elif probability < high:
                risk_tier = "Moderate"
            else:
                risk_tier = "High"

            # SHAP
            shap_output = EXPLAINER.shap_values(
                input_df
            )

            if isinstance(shap_output, list):
                shap_values = (
                    shap_output[1]
                    if len(shap_output) > 1
                    else shap_output[0]
                )
            else:
                shap_values = shap_output

            shap_values = np.asarray(
                shap_values
            ).squeeze()

            if shap_values.ndim > 1:
                shap_values = shap_values.reshape(-1)

            if len(shap_values) > len(FEATURES):
                shap_values = shap_values[:len(FEATURES)]

            if len(shap_values) != len(FEATURES):
                raise ValueError(
                    f"SHAP output mismatch: expected {len(FEATURES)}, "
                    f"received {len(shap_values)}."
                )

            explanation = pd.DataFrame(
                {
                    "Feature": [
                        DISPLAY.get(f, f)
                        for f in FEATURES
                    ],
                    "Feature Code": FEATURES,
                    "Patient Value": [
                        input_df.iloc[0][f]
                        for f in FEATURES
                    ],
                    "SHAP Value": shap_values,
                    "Absolute SHAP": np.abs(shap_values),
                }
            ).sort_values(
                "Absolute SHAP",
                ascending=False,
            ).reset_index(drop=True)

            explanation["Direction"] = np.where(
                explanation["SHAP Value"] >= 0,
                "Increases CKD model output",
                "Decreases CKD model output",
            )

            st.session_state.last_result = {
                "input": input_df,
                "probability": probability,
                "prediction": prediction,
                "risk_tier": risk_tier,
                "explanation": explanation,
            }

            result = st.session_state.last_result

        except Exception as e:

            st.error("Unable to complete the assessment.")
            st.exception(e)
            result = None

    else:
        result = st.session_state.get("last_result")


    # ========================================================
    # RESULT DASHBOARD
    # ========================================================

    if result is not None:

        st.markdown("---")

        st.markdown(
            '<div class="section-title">Assessment Result</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Prediction",
                result["prediction"],
            )

        with c2:
            st.metric(
                "CKD Probability",
                f"{result['probability'] * 100:.2f}%",
            )

        with c3:
            st.metric(
                "Research Risk Tier",
                result["risk_tier"],
            )

        st.progress(
            min(max(result["probability"], 0.0), 1.0),
            text=(
                f"Model probability · "
                f"{result['probability'] * 100:.2f}%"
            ),
        )

        tier = result["risk_tier"]

        if tier == "Low":
            css_class = "status-low"
            message = (
                "The model assigns this patient input to the "
                "Low research risk tier."
            )
        elif tier == "Moderate":
            css_class = "status-moderate"
            message = (
                "The model assigns this patient input to the "
                "Moderate research risk tier."
            )
        else:
            css_class = "status-high"
            message = (
                "The model assigns this patient input to the "
                "High research risk tier."
            )

        st.markdown(
            f'<div class="{css_class}">● &nbsp;{message}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        st.markdown(
            '<div class="section-title">Top Contributing Factors</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "SHAP values explain how features contribute to the model "
            "output. They do not establish clinical causation."
        )

        top = result["explanation"].head(8).copy()

        st.dataframe(
            top[
                [
                    "Feature",
                    "Patient Value",
                    "SHAP Value",
                    "Direction",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        downloadable = {
            "prediction": result["prediction"],
            "ckd_probability": result["probability"],
            "risk_tier": result["risk_tier"],
            "model": "XGBoost",
            "explainability": "SHAP TreeExplainer",
            "patient_input": {
                feature: result["input"].iloc[0][feature]
                for feature in FEATURES
            },
            "top_contributing_features": top[
                [
                    "Feature",
                    "Feature Code",
                    "Patient Value",
                    "SHAP Value",
                    "Direction",
                ]
            ].to_dict(orient="records"),
            "note": "Research/screening prototype; not a medical diagnosis.",
        }

        st.download_button(
            "⬇️  Download Assessment JSON",
            data=json.dumps(
                downloadable,
                indent=2,
                default=str,
            ),
            file_name="ckd_assessment_result.json",
            mime="application/json",
        )


# ============================================================
# EXPLAINABILITY
# ============================================================

with tab_explainability:

    st.markdown(
        '<div class="section-title">Explainable AI Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'SHAP provides global feature importance and patient-specific '
        'feature attribution for the XGBoost prediction.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Global Feature Importance")

    try:

        if isinstance(SHAP_IMPORTANCE, list) and SHAP_IMPORTANCE:

            importance_df = pd.DataFrame(
                SHAP_IMPORTANCE
            )

            feature_col = next(
                (
                    c for c in
                    ["feature", "Feature", "feature_name"]
                    if c in importance_df.columns
                ),
                None,
            )

            importance_col = next(
                (
                    c for c in
                    [
                        "mean_abs_shap",
                        "Mean |SHAP|",
                        "importance",
                        "mean_abs",
                    ]
                    if c in importance_df.columns
                ),
                None,
            )

            if feature_col and importance_col:

                importance_df["Display Feature"] = (
                    importance_df[feature_col].map(
                        lambda x: DISPLAY.get(x, x)
                    )
                )

                importance_df["Importance"] = pd.to_numeric(
                    importance_df[importance_col],
                    errors="coerce",
                )

                importance_df = importance_df.sort_values(
                    "Importance",
                    ascending=False,
                )

                chart_data = (
                    importance_df
                    .head(12)
                    .set_index("Display Feature")["Importance"]
                )

                st.bar_chart(chart_data)

                st.dataframe(
                    importance_df[
                        [
                            "Display Feature",
                            feature_col,
                            "Importance",
                        ]
                    ].rename(
                        columns={
                            "Display Feature": "Feature",
                            feature_col: "Feature Code",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.info(
                    "The SHAP importance file has an unexpected format."
                )

        else:
            st.info(
                "Global SHAP importance is not available."
            )

    except Exception as e:

        st.warning(
            "Unable to display global SHAP importance."
        )
        st.exception(e)


    st.markdown("#### Patient-Level Explanation")

    result = st.session_state.get("last_result")

    if result is not None:

        patient_shap = (
            result["explanation"]
            .head(10)
            .copy()
            .sort_values("SHAP Value")
        )

        st.bar_chart(
            patient_shap.set_index("Feature")["SHAP Value"]
        )

    else:

        st.info(
            "Run a patient assessment to generate a patient-level "
            "SHAP explanation."
        )


# ============================================================
# PERFORMANCE
# ============================================================

with tab_performance:

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Performance values correspond to the trained model artifact '
        'used by this application.'
        '</div>',
        unsafe_allow_html=True,
    )

    accuracy = METRICS.get("accuracy")
    precision = METRICS.get("precision")
    recall = METRICS.get("recall")
    f1 = METRICS.get("f1")
    auc = METRICS.get("auc")

    c1, c2, c3, c4, c5 = st.columns(5)

    if accuracy is not None:
        c1.metric(
            "Accuracy",
            f"{float(accuracy) * 100:.2f}%",
        )

    if precision is not None:
        c2.metric(
            "Precision",
            f"{float(precision) * 100:.2f}%",
        )

    if recall is not None:
        c3.metric(
            "Recall",
            f"{float(recall) * 100:.2f}%",
        )

    if f1 is not None:
        c4.metric(
            "F1 Score",
            f"{float(f1) * 100:.2f}%",
        )

    if auc is not None:
        c5.metric(
            "ROC-AUC",
            f"{float(auc):.3f}",
        )

    st.markdown("#### Confusion Matrix")

    cm = METRICS.get("confusion_matrix")

    if cm is not None:

        try:

            cm_df = pd.DataFrame(
                np.asarray(cm),
                index=[
                    "Actual Not CKD",
                    "Actual CKD",
                ],
                columns=[
                    "Predicted Not CKD",
                    "Predicted CKD",
                ],
            )

            st.dataframe(
                cm_df,
                use_container_width=True,
            )

        except Exception:
            st.write(cm)

    st.markdown("#### Dataset Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Records",
        len(DATASET),
    )

    c2.metric(
        "Clinical Features",
        len(FEATURES),
    )

    if "class" in DATASET.columns:
        c3.metric(
            "Classes",
            DATASET["class"].nunique(),
        )


# ============================================================
# RESEARCH DETAILS
# ============================================================

with tab_research:

    st.markdown(
        '<div class="section-title">Research Framework</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
        <b>Research Title</b><br>
        Explainable AI-Based Framework for Early Detection and Risk
        Assessment of Chronic Kidney Disease Using Machine Learning
        <br><br>

        <b>Framework</b><br>
        Clinical Data → Preprocessing → XGBoost → CKD Probability →
        Risk Assessment → SHAP Explanation
        <br><br>

        <b>Model</b><br>
        XGBoost
        <br><br>

        <b>Explainability</b><br>
        SHAP TreeExplainer
        <br><br>

        <b>Clinical Inputs</b><br>
        24 research-dataset attributes
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    st.markdown("#### Research Risk Thresholds")

    moderate = float(
        RISK_THRESHOLDS.get("moderate", 0.33)
    )
    high = float(
        RISK_THRESHOLDS.get("high", 0.66)
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Low",
        f"< {moderate:.2f}",
    )

    c2.metric(
        "Moderate",
        f"{moderate:.2f} – < {high:.2f}",
    )

    c3.metric(
        "High",
        f"≥ {high:.2f}",
    )

    st.markdown(
        """
        <div class="medical-note">
        <b>Research limitation:</b>
        The risk thresholds shown above are design thresholds for
        this research prototype and are not clinically calibrated.
        A model prediction should not be interpreted as a diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        CKD-AI Research Prototype &nbsp;•&nbsp;
        XGBoost + SHAP &nbsp;•&nbsp;
        Explainable CKD Risk Assessment
    </div>
    """,
    unsafe_allow_html=True,
)
