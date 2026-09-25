import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CKD-AI | Explainable CKD Risk Assessment",
    page_icon="🩺",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
DATASET_PATH = BASE / "ckd_dataset.csv"


# ============================================================
# LOAD MODEL ARTIFACT
# ============================================================

MODEL_PATH = ART / "ckd_model.joblib"
METADATA_PATH = ART / "metadata.json"
METRICS_PATH = ART / "metrics.json"
SHAP_PATH = ART / "shap_importance.json"


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please make sure the GitHub repository contains:\n\n"
        "artifacts/ckd_model.joblib"
    )
    st.stop()


if not METADATA_PATH.exists():
    st.error(
        "metadata.json not found. Please make sure the GitHub repository contains:\n\n"
        "artifacts/metadata.json"
    )
    st.stop()


if not METRICS_PATH.exists():
    st.error(
        "metrics.json not found. Please make sure the GitHub repository contains:\n\n"
        "artifacts/metrics.json"
    )
    st.stop()


if not SHAP_PATH.exists():
    st.error(
        "shap_importance.json not found. Please make sure the GitHub repository contains:\n\n"
        "artifacts/shap_importance.json"
    )
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:
    bundle = joblib.load(MODEL_PATH)

    MODEL = bundle["model"]
    FEATURES = bundle["features"]
    THRESHOLDS = bundle["risk_thresholds"]

except Exception as e:
    st.error("Unable to load the trained CKD model.")
    st.exception(e)
    st.stop()


# ============================================================
# LOAD METADATA
# ============================================================

try:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        META = json.load(f)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        METRICS = json.load(f)

    with open(SHAP_PATH, "r", encoding="utf-8") as f:
        SHAP_IMPORTANCE = json.load(f)

except Exception as e:
    st.error("Unable to load model metadata or evaluation results.")
    st.exception(e)
    st.stop()


# ============================================================
# LOAD DATASET FOR DEFAULT VALUES
# ============================================================

if not DATASET_PATH.exists():
    st.error(
        "ckd_dataset.csv not found. Please make sure the dataset is present "
        "in the root of your GitHub repository."
    )
    st.stop()


try:
    DATASET = pd.read_csv(DATASET_PATH)
except Exception as e:
    st.error("Unable to read ckd_dataset.csv.")
    st.exception(e)
    st.stop()


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

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


# ============================================================
# BINARY FEATURES
# ============================================================

# These fields use 0/1 encoding in the supplied research dataset.
binary_features = {
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
}


# Numeric categorical features.
categorical_numeric_features = {
    "sg",
    "al",
    "su",
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


# ============================================================
# DEFAULT VALUES
# ============================================================

def default_value(col):
    """
    Return the median value of the corresponding feature
    from the supplied research dataset.
    """

    if col not in DATASET.columns:
        return 0.0

    series = pd.to_numeric(DATASET[col], errors="coerce")

    if series.dropna().empty:
        return 0.0

    return float(series.median())


def get_default(col):
    """
    Generate a safe default value for Streamlit input widgets.
    """

    value = default_value(col)

    if col in binary_features:
        return int(round(value))

    if col in categorical_numeric_features:
        if col == "sg":
            return float(value)

        return int(round(value))

    return float(value)


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def get_explainer(model):
    return shap.TreeExplainer(model)


try:
    EXPLAINER = get_explainer(MODEL)
except Exception as e:
    st.error("Unable to initialize the SHAP explainer.")
    st.exception(e)
    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        border: 1px solid #d9dee7;
        border-radius: 12px;
        padding: 16px;
        background: #ffffff;
    }

    .warning-box {
        padding: 12px;
        border-radius: 10px;
        background: #fff4e5;
        border: 1px solid #f0c36d;
    }

    .disclaimer {
        padding: 12px;
        border-radius: 10px;
        background: #f4f6f8;
        border: 1px solid #d7dce2;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🩺 CKD-AI")

st.subheader(
    "Explainable AI-Based Chronic Kidney Disease Risk Assessment"
)

st.caption(
    "Research prototype based on the supplied CKD dataset and "
    "XGBoost + SHAP pipeline."
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

    <b>Research / screening prototype:</b>

    This application is not a medical diagnostic device and does not
    replace clinician assessment, eGFR/ACR evaluation, laboratory
    confirmation, or professional medical advice.

    Risk thresholds are research-design thresholds and are not
    clinically calibrated cut-offs.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab_assess, tab_explain, tab_perf, tab_about = st.tabs(
    [
        "🧪 Patient Assessment",
        "🔎 Explainability",
        "📊 Model Performance",
        "ℹ️ Research Details",
    ]
)


# ============================================================
# TAB 1 — PATIENT ASSESSMENT
# ============================================================

with tab_assess:

    st.markdown("### Enter the 24 clinical attributes")

    st.caption(
        "The model expects the same numeric encoding used in the supplied "
        "research dataset. No identifying information is required."
    )

    values = {}


    # --------------------------------------------------------
    # BASIC MEASUREMENTS
    # --------------------------------------------------------

    st.markdown("#### Basic measurements")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        values["age"] = st.number_input(
            "Age",
            min_value=0.0,
            value=max(0.0, get_default("age")),
            step=1.0,
        )

    with c2:
        values["bp"] = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            value=max(0.0, get_default("bp")),
            step=1.0,
        )

    with c3:
        values["sg"] = st.number_input(
            "Specific Gravity",
            min_value=0.0,
            value=max(0.0, get_default("sg")),
            step=0.001,
            format="%.3f",
        )

    with c4:
        values["al"] = st.number_input(
            "Albumin",
            min_value=0.0,
            value=max(0.0, get_default("al")),
            step=1.0,
        )


    # --------------------------------------------------------
    # URINE ATTRIBUTES
    # --------------------------------------------------------

    st.markdown("#### Urine-related attributes")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        values["su"] = st.number_input(
            "Sugar",
            min_value=0.0,
            value=max(0.0, get_default("su")),
            step=1.0,
        )

    with c2:
        values["rbc"] = st.selectbox(
            "Red Blood Cells (0/1)",
            [0, 1],
            index=int(get_default("rbc")),
            help=HELP["rbc"],
        )

    with c3:
        values["pc"] = st.selectbox(
            "Pus Cell (0/1)",
            [0, 1],
            index=int(get_default("pc")),
            help=HELP["pc"],
        )

    with c4:
        values["pcc"] = st.selectbox(
            "Pus Cell Clumps (0/1)",
            [0, 1],
            index=int(get_default("pcc")),
            help=HELP["pcc"],
        )

    with c5:
        values["ba"] = st.selectbox(
            "Bacteria (0/1)",
            [0, 1],
            index=int(get_default("ba")),
            help=HELP["ba"],
        )


    # --------------------------------------------------------
    # BLOOD TEST ATTRIBUTES
    # --------------------------------------------------------

    st.markdown("#### Blood-test attributes")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        values["bgr"] = st.number_input(
            "Blood Glucose Random",
            min_value=0.0,
            value=max(0.0, get_default("bgr")),
            step=1.0,
        )

    with c2:
        values["bu"] = st.number_input(
            "Blood Urea",
            min_value=0.0,
            value=max(0.0, get_default("bu")),
            step=1.0,
        )

    with c3:
        values["sc"] = st.number_input(
            "Serum Creatinine",
            min_value=0.0,
            value=max(0.0, get_default("sc")),
            step=0.1,
        )

    with c4:
        values["sod"] = st.number_input(
            "Sodium",
            min_value=0.0,
            value=max(0.0, get_default("sod")),
            step=0.1,
        )


    # --------------------------------------------------------
    # ADDITIONAL BLOOD PARAMETERS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        values["pot"] = st.number_input(
            "Potassium",
            min_value=0.0,
            value=max(0.0, get_default("pot")),
            step=0.1,
        )

    with c2:
        values["hemo"] = st.number_input(
            "Haemoglobin",
            min_value=0.0,
            value=max(0.0, get_default("hemo")),
            step=0.1,
        )

    with c3:
        values["pcv"] = st.number_input(
            "Packed Cell Volume",
            min_value=0.0,
            value=max(0.0, get_default("pcv")),
            step=0.1,
        )

    with c4:
        values["wc"] = st.number_input(
            "White Blood Cell Count",
            min_value=0.0,
            value=max(0.0, get_default("wc")),
            step=10.0,
        )


    # --------------------------------------------------------
    # RED BLOOD CELL COUNT
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:
        values["rc"] = st.number_input(
            "Red Blood Cell Count",
            min_value=0.0,
            value=max(0.0, get_default("rc")),
            step=0.1,
        )

    with c2:
        st.info(
            "Use the same measurement units and encoding conventions "
            "as the research dataset."
        )


    # --------------------------------------------------------
    # MEDICAL CONDITION FLAGS
    # --------------------------------------------------------

    st.markdown("#### Medical-condition flags")

    c1, c2, c3, c4, c5 = st.columns(5)

    medical_flags = [
        "htn",
        "dm",
        "cad",
        "appet",
        "pe",
    ]

    for widget_col, col in zip(
        [c1, c2, c3, c4, c5],
        medical_flags,
    ):

        with widget_col:

            values[col] = st.selectbox(
                DISPLAY[col] + " (0/1)",
                [0, 1],
                index=int(get_default(col)),
                help=HELP[col],
            )


    c1, c2 = st.columns(2)

    with c1:

        values["ane"] = st.selectbox(
            "Anaemia (0/1)",
            [0, 1],
            index=int(get_default("ane")),
            help=HELP["ane"],
        )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    st.divider()

    analyze = st.button(
        "🔍 Analyze CKD Risk",
        type="primary",
        use_container_width=True,
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    if analyze:

        try:

            # Ensure features are in exactly the same order
            # used during model training.
            row = pd.DataFrame(
                [[values[c] for c in FEATURES]],
                columns=FEATURES,
            )


            # ------------------------------------------------
            # DATASET RANGE CHECK
            # ------------------------------------------------

            outside = []

            for c in FEATURES:

                if c not in META["dataset_ranges"]:
                    continue

                lo = META["dataset_ranges"][c]["min"]
                hi = META["dataset_ranges"][c]["max"]

                val = float(row.iloc[0][c])

                if val < lo or val > hi:

                    outside.append(
                        f"{DISPLAY.get(c, c)} "
                        f"({val:g}; dataset range {lo:g}–{hi:g})"
                    )


            if outside:

                st.warning(
                    "One or more values are outside the empirical "
                    "range of the supplied training dataset. "
                    "The app will still send the values to the model "
                    "without clipping. This may reduce reliability.\n\n"
                    + "; ".join(outside[:8])
                )


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            probability_array = MODEL.predict_proba(row)

            p = float(probability_array[:, 1][0])


            # Binary prediction
            label = (
                "CKD Positive"
                if p >= 0.50
                else "Not CKD"
            )


            # ------------------------------------------------
            # RISK TIER
            # ------------------------------------------------

            if p < THRESHOLDS["moderate"]:

                tier = "Low"

            elif p < THRESHOLDS["high"]:

                tier = "Moderate"

            else:

                tier = "High"


            # ------------------------------------------------
            # SHAP EXPLANATION
            # ------------------------------------------------

            shap_values = EXPLAINER.shap_values(row)


            # Handle different SHAP output formats.
            if isinstance(shap_values, list):

                if len(shap_values) > 1:
                    shap_values = shap_values[1]
                else:
                    shap_values = shap_values[0]


            shap_values = np.asarray(shap_values)


            # Remove unnecessary dimensions.
            shap_values = np.squeeze(shap_values)


            # Make sure we have one value per feature.
            if shap_values.ndim != 1:

                shap_values = shap_values.reshape(-1)


            # Some SHAP/XGBoost versions may return
            # one extra value for the base output.
            if len(shap_values) != len(FEATURES):

                shap_values = shap_values[:len(FEATURES)]


            # ------------------------------------------------
            # CREATE EXPLANATION TABLE
            # ------------------------------------------------

            explanation = pd.DataFrame(
                {
                    "Feature": [
                        DISPLAY.get(c, c)
                        for c in FEATURES
                    ],

                    "Feature_Code": FEATURES,

                    "SHAP": shap_values,

                    "Absolute_SHAP": np.abs(
                        shap_values
                    ),

                    "Value": [
                        float(row.iloc[0][c])
                        for c in FEATURES
                    ],
                }
            ).sort_values(
                "Absolute_SHAP",
                ascending=False,
            )


            # ------------------------------------------------
            # SAVE RESULT IN SESSION
            # ------------------------------------------------

            st.session_state["last_result"] = {

                "row": row,

                "probability": p,

                "label": label,

                "tier": tier,

                "explanation": explanation,
            }


        except Exception as e:

            st.error(
                "An error occurred while analyzing the patient data."
            )

            st.exception(e)


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if "last_result" in st.session_state:

        r = st.session_state["last_result"]


        st.markdown("### Assessment Result")


        m1, m2, m3 = st.columns(3)


        m1.metric(
            "Prediction",
            r["label"],
        )


        m2.metric(
            "CKD Probability",
            f"{r['probability'] * 100:.1f}%",
        )


        m3.metric(
            "Risk Tier",
            r["tier"],
        )


        st.progress(
            min(
                max(
                    r["probability"],
                    0.0,
                ),
                1.0,
            ),
            text=(
                f"Model probability: "
                f"{r['probability'] * 100:.1f}%"
            ),
        )


        # ----------------------------------------------------
        # RISK INTERPRETATION
        # ----------------------------------------------------

        if r["tier"] == "Low":

            st.info(
                "Low research risk tier based on the model's "
                "predicted probability."
            )

        elif r["tier"] == "Moderate":

            st.warning(
                "Moderate research risk tier based on the "
                "model's predicted probability."
            )

        else:

            st.error(
                "High research risk tier based on the "
                "model's predicted probability."
            )


        # ----------------------------------------------------
        # TOP CONTRIBUTING FACTORS
        # ----------------------------------------------------

        st.markdown("#### Top contributing factors")


        top = (
            r["explanation"]
            .head(8)
            .copy()
        )


        top["Direction"] = np.where(
            top["SHAP"] >= 0,
            "Increases CKD model output",
            "Decreases CKD model output",
        )


        top["Impact"] = (
            top["SHAP"]
            .abs()
            .round(4)
        )


        st.dataframe(
            top[
                [
                    "Feature",
                    "Value",
                    "SHAP",
                    "Direction",
                    "Impact",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


        # ----------------------------------------------------
        # DOWNLOAD RESULT
        # ----------------------------------------------------

        result_json = {

            "prediction": r["label"],

            "ckd_probability": r["probability"],

            "risk_tier": r["tier"],

            "top_factors": (
                top[
                    [
                        "Feature",
                        "Value",
                        "SHAP",
                        "Direction",
                    ]
                ].to_dict(
                    orient="records"
                )
            ),

            "model": "XGBoost",

            "research_note": (
                "Prototype screening output; "
                "not a medical diagnosis."
            ),
        }


        st.download_button(

            "⬇️ Download Assessment JSON",

            data=json.dumps(
                result_json,
                indent=2,
            ),

            file_name="ckd_assessment_result.json",

            mime="application/json",
        )


# ============================================================
# TAB 2 — EXPLAINABILITY
# ============================================================

with tab_explain:

    st.markdown("### Explainability")

    st.write(
        "The deployed XGBoost model is explained using "
        "SHAP TreeExplainer, consistent with the research design."
    )


    # --------------------------------------------------------
    # GLOBAL SHAP IMPORTANCE
    # --------------------------------------------------------

    st.markdown("#### Global feature importance")


    imp = pd.DataFrame(
        SHAP_IMPORTANCE
    )


    if not imp.empty:

        if "feature" in imp.columns:

            imp["Feature"] = imp[
                "feature"
            ].map(
                lambda x: DISPLAY.get(
                    x,
                    x,
                )
            )

        if "mean_abs_shap" in imp.columns:

            imp = imp.rename(
                columns={
                    "mean_abs_shap":
                        "Mean |SHAP|",

                    "feature":
                        "Feature Code",
                }
            )


        if (
            "Feature" in imp.columns
            and "Mean |SHAP|" in imp.columns
        ):

            chart_data = (
                imp.head(12)
                .set_index("Feature")[
                    "Mean |SHAP|"
                ]
            )

            st.bar_chart(
                chart_data
            )


        display_columns = []

        if "Feature" in imp.columns:
            display_columns.append("Feature")

        if "Feature Code" in imp.columns:
            display_columns.append(
                "Feature Code"
            )

        if "Mean |SHAP|" in imp.columns:
            display_columns.append(
                "Mean |SHAP|"
            )


        if display_columns:

            st.dataframe(
                imp[
                    display_columns
                ],
                use_container_width=True,
                hide_index=True,
            )


    # --------------------------------------------------------
    # PATIENT LEVEL SHAP
    # --------------------------------------------------------

    if "last_result" in st.session_state:

        st.markdown(
            "#### Latest patient-level explanation"
        )


        e = (
            st.session_state[
                "last_result"
            ]["explanation"]
            .head(10)
            .copy()
        )


        e = e.sort_values(
            "SHAP"
        )


        e = e.set_index(
            "Feature"
        )["SHAP"]


        st.bar_chart(e)


    else:

        st.info(
            "Run an assessment first to display "
            "the patient-level SHAP explanation."
        )


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================

with tab_perf:

    st.markdown("### Model Performance")

    st.caption(
        "These values are from the model artifact trained "
        "from the supplied dataset in this application build."
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Accuracy",
        f"{METRICS['accuracy'] * 100:.2f}%",
    )


    c2.metric(
        "Precision",
        f"{METRICS['precision'] * 100:.2f}%",
    )


    c3.metric(
        "Recall",
        f"{METRICS['recall'] * 100:.2f}%",
    )


    c4.metric(
        "AUC",
        f"{METRICS['auc']:.3f}",
    )


    # --------------------------------------------------------
    # F1 SCORE
    # --------------------------------------------------------

    if "f1" in METRICS:

        st.metric(
            "F1 Score",
            f"{METRICS['f1'] * 100:.2f}%",
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = np.array(
        METRICS["confusion_matrix"]
    )


    st.markdown(
        "#### XGBoost confusion matrix"
    )


    cm_df = pd.DataFrame(

        cm,

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


    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.markdown("#### Dataset")


    d1, d2, d3 = st.columns(3)


    d1.metric(
        "Records",
        META["dataset_shape"][0],
    )


    d2.metric(
        "Features",
        len(FEATURES),
    )


    d3.metric(
        "CKD / Not CKD",
        (
            f"{META['class_distribution']['CKD']} / "
            f"{META['class_distribution']['Not CKD']}"
        ),
    )


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    if "n_train" in METRICS:

        st.write(
            f"Training records: "
            f"**{METRICS['n_train']}**"
        )


    if "n_test" in METRICS:

        st.write(
            f"Testing records: "
            f"**{METRICS['n_test']}**"
        )


# ============================================================
# TAB 4 — RESEARCH DETAILS
# ============================================================

with tab_about:

    st.markdown("### Research Implementation")


    st.markdown(
        """
        **Module 1 — Preprocessing & Feature Engineering**

        - 24 clinical attributes
        - Input validation
        - Dataset-consistent numeric/binary encoding


        **Module 2 — Prediction**

        - XGBoost deployment model
        - 250 trees
        - Maximum depth 4
        - Learning rate 0.08


        **Module 3 — Explainability & Risk Assessment**

        - SHAP TreeExplainer
        - CKD probability
        - Low / Moderate / High research risk tier
        - Ranked contributing features
        """
    )


    # --------------------------------------------------------
    # RESEARCH DATASET
    # --------------------------------------------------------

    st.markdown("### Research dataset")


    st.write(
        "400 records; 248 CKD and 152 non-CKD cases. "
        "The application does not collect or store names, "
        "national IDs, or other direct identifiers."
    )


    # --------------------------------------------------------
    # RISK THRESHOLDS
    # --------------------------------------------------------

    st.markdown("### Research risk thresholds")


    low_threshold = 0.33
    high_threshold = 0.66


    if "moderate" in THRESHOLDS:

        low_threshold = THRESHOLDS["moderate"]


    if "high" in THRESHOLDS:

        high_threshold = THRESHOLDS["high"]


    t1, t2, t3 = st.columns(3)


    t1.metric(
        "Low",
        f"< {low_threshold:.2f}",
    )


    t2.metric(
        "Moderate",
        f"{low_threshold:.2f} – < {high_threshold:.2f}",
    )


    t3.metric(
        "High",
        f"≥ {high_threshold:.2f}",
    )


    # --------------------------------------------------------
    # IMPORTANT WARNING
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="warning-box">

        <b>Important:</b>

        The supplied research report states that the working dataset
        is a single-context, 400-record research dataset and that the
        risk thresholds are not clinically calibrated.

        This application therefore demonstrates the research prototype
        and should not be used to diagnose or treat a patient.

        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.markdown("### Model information")


    st.write(
        "Deployment model: **XGBoost**"
    )


    st.write(
        "Explainability method: **SHAP TreeExplainer**"
    )


    st.write(
        "Input variables: **24 clinical attributes**"
    )


    st.write(
        "Risk assessment: **Low / Moderate / High**"
    )


    st.write(
        "Prediction threshold: **0.50 for binary classification**"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CKD-AI Research Prototype | "
    "Explainable AI-Based Early CKD Detection and Risk Assessment"
)
