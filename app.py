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

MODEL_PATH = ART / "ckd_model.joblib"
METADATA_PATH = ART / "metadata.json"
METRICS_PATH = ART / "metrics.json"
SHAP_PATH = ART / "shap_importance.json"
DATASET_PATH = BASE / "ckd_dataset.csv"


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURES = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
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


# ============================================================
# LOAD REQUIRED FILES
# ============================================================

required_files = {
    "Model": MODEL_PATH,
    "Metadata": METADATA_PATH,
    "Metrics": METRICS_PATH,
    "SHAP importance": SHAP_PATH,
    "Dataset": DATASET_PATH,
}

missing_files = [
    f"{name}: {path}"
    for name, path in required_files.items()
    if not path.exists()
]

if missing_files:
    st.error("Required project files are missing:")
    for item in missing_files:
        st.write(f"- `{item}`")
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    bundle = joblib.load(MODEL_PATH)

    MODEL = bundle["model"]

    MODEL_FEATURES = bundle.get(
        "features",
        FEATURES,
    )

    THRESHOLDS = bundle.get(
        "risk_thresholds",
        {
            "moderate": 0.33,
            "high": 0.66,
        },
    )

except Exception as e:

    st.error("Unable to load the trained XGBoost model.")
    st.exception(e)
    st.stop()


# Make sure the model feature order is used.
if MODEL_FEATURES:
    FEATURES = list(MODEL_FEATURES)


# ============================================================
# LOAD METADATA
# ============================================================

try:

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        META = json.load(f)

except Exception as e:

    st.warning(
        "metadata.json could not be fully loaded. "
        "The application will continue with default settings."
    )

    META = {}


# ============================================================
# LOAD METRICS
# ============================================================

try:

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        METRICS = json.load(f)

except Exception:

    METRICS = {}


# ============================================================
# LOAD SHAP GLOBAL IMPORTANCE
# ============================================================

try:

    with open(
        SHAP_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        SHAP_IMPORTANCE = json.load(f)

except Exception:

    SHAP_IMPORTANCE = []


# ============================================================
# LOAD DATASET
# ============================================================

try:

    DATASET = pd.read_csv(DATASET_PATH)

except Exception as e:

    st.error("Unable to read ckd_dataset.csv.")
    st.exception(e)
    st.stop()


# ============================================================
# SHAP EXPLAINER
# ============================================================

try:

    # IMPORTANT:
    # Do NOT use @st.cache_resource with the XGBoost model
    # because Streamlit may fail while hashing XGBClassifier.
    EXPLAINER = shap.TreeExplainer(MODEL)

except Exception as e:

    st.error("Unable to initialize the SHAP explainer.")
    st.exception(e)
    st.stop()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .disclaimer {
        padding: 14px;
        border-radius: 10px;
        background: #f4f6f8;
        border: 1px solid #d7dce2;
        font-size: 0.92rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title("🩺 CKD-AI")

st.subheader(
    "Explainable AI-Based Early Detection and Risk Assessment "
    "of Chronic Kidney Disease"
)

st.caption(
    "XGBoost-based CKD prediction with SHAP explainability"
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

    <b>Research / screening prototype</b><br><br>

    This application is intended for research and screening
    demonstration only. It is not a medical diagnostic device
    and should not replace professional medical evaluation,
    laboratory testing, eGFR/ACR assessment, or clinical judgment.

    The displayed risk thresholds are research-design thresholds
    and are not clinically calibrated.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab_assessment, tab_explain, tab_performance, tab_research = st.tabs(
    [
        "🧪 Patient Assessment",
        "🔎 Explainability",
        "📊 Model Performance",
        "📚 Research Details",
    ]
)


# ============================================================
# TAB 1 — PATIENT ASSESSMENT
# ============================================================

with tab_assessment:

    st.markdown("### Patient Clinical Assessment")

    st.info(
        "Enter all 24 clinical attributes. "
        "The application intentionally starts with blank fields "
        "so that the prediction is based only on the values you provide."
    )


    # ========================================================
    # SESSION STATE
    # ========================================================

    if "last_result" not in st.session_state:
        st.session_state.last_result = None


    # ========================================================
    # BASIC MEASUREMENTS
    # ========================================================

    st.markdown("#### 1. Basic Measurements")

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
            placeholder="Enter blood pressure",
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


    # ========================================================
    # URINE PARAMETERS
    # ========================================================

    st.markdown("#### 2. Urine Parameters")

    c1, c2, c3 = st.columns(3)

    with c1:

        al = st.selectbox(
            "Albumin",
            options=[0, 1, 2, 3, 4, 5],
            index=None,
            placeholder="Select Albumin value",
            help=(
                "Ordinal dataset encoding. "
                "Select the value corresponding to the patient's record."
            ),
        )

    with c2:

        su = st.selectbox(
            "Sugar",
            options=[0, 1, 2, 3, 4, 5],
            index=None,
            placeholder="Select Sugar value",
            help="Ordinal dataset encoding.",
        )

    with c3:

        rbc = st.selectbox(
            "Red Blood Cells",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    c1, c2, c3 = st.columns(3)

    with c1:

        pc = st.selectbox(
            "Pus Cell",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    with c2:

        pcc = st.selectbox(
            "Pus Cell Clumps",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    with c3:

        ba = st.selectbox(
            "Bacteria",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    # ========================================================
    # BLOOD PARAMETERS
    # ========================================================

    st.markdown("#### 3. Blood Parameters")

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


    # ========================================================
    # MEDICAL CONDITIONS
    # ========================================================

    st.markdown("#### 4. Medical Condition Indicators")

    c1, c2, c3 = st.columns(3)

    with c1:

        htn = st.selectbox(
            "Hypertension",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    with c2:

        dm = st.selectbox(
            "Diabetes Mellitus",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    with c3:

        cad = st.selectbox(
            "Coronary Artery Disease",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    c1, c2, c3 = st.columns(3)

    with c1:

        appet = st.selectbox(
            "Appetite",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    with c2:

        pe = st.selectbox(
            "Pedal Edema",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )

    with c3:

        ane = st.selectbox(
            "Anaemia",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    # ========================================================
    # COLLECT INPUTS
    # ========================================================

    values = {
        "age": age,
        "bp": bp,
        "sg": sg,
        "al": al,
        "su": su,
        "rbc": rbc,
        "pc": pc,
        "pcc": pcc,
        "ba": ba,
        "bgr": bgr,
        "bu": bu,
        "sc": sc,
        "sod": sod,
        "pot": pot,
        "hemo": hemo,
        "pcv": pcv,
        "wc": wc,
        "rc": rc,
        "htn": htn,
        "dm": dm,
        "cad": cad,
        "appet": appet,
        "pe": pe,
        "ane": ane,
    }


    # ========================================================
    # VALIDATION
    # ========================================================

    missing_fields = [
        DISPLAY.get(key, key)
        for key, value in values.items()
        if value is None
    ]


    if missing_fields:

        st.warning(
            f"Please enter/select all 24 clinical attributes. "
            f"{len(missing_fields)} field(s) are still missing."
        )

        with st.expander("Missing fields"):

            for field in missing_fields:
                st.write(f"• {field}")


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.divider()

    analyze = st.button(
        "🔍 Analyze CKD Risk",
        type="primary",
        use_container_width=True,
        disabled=len(missing_fields) > 0,
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if analyze:

        try:

            # -----------------------------------------------
            # Create model input
            # -----------------------------------------------

            input_values = []

            for feature in FEATURES:

                value = values[feature]

                if feature in BINARY_FEATURES:
                    value = int(value)

                elif feature in {"al", "su"}:
                    value = int(value)

                else:
                    value = float(value)

                input_values.append(value)


            row = pd.DataFrame(
                [input_values],
                columns=FEATURES,
            )


            # -----------------------------------------------
            # Dataset range warning
            # -----------------------------------------------

            outside_range = []


            dataset_ranges = META.get(
                "dataset_ranges",
                {},
            )


            for feature in FEATURES:

                if feature not in dataset_ranges:
                    continue

                try:

                    lower = float(
                        dataset_ranges[feature]["min"]
                    )

                    upper = float(
                        dataset_ranges[feature]["max"]
                    )

                    value = float(
                        row.iloc[0][feature]
                    )

                    if value < lower or value > upper:

                        outside_range.append(
                            f"{DISPLAY.get(feature, feature)} "
                            f"= {value:g} "
                            f"(dataset range: "
                            f"{lower:g}–{upper:g})"
                        )

                except Exception:
                    pass


            if outside_range:

                st.warning(
                    "Some values are outside the empirical "
                    "range of the supplied research dataset. "
                    "The model will receive the values without "
                    "clipping them."
                )

                with st.expander(
                    "Values outside the training-data range"
                ):

                    for item in outside_range:
                        st.write(f"• {item}")


            # -----------------------------------------------
            # Model probability
            # -----------------------------------------------

            probability = MODEL.predict_proba(row)

            p = float(
                probability[0][1]
            )


            # -----------------------------------------------
            # Binary prediction
            # -----------------------------------------------

            prediction = (
                "CKD Positive"
                if p >= 0.50
                else "Not CKD"
            )


            # -----------------------------------------------
            # Research risk tier
            # -----------------------------------------------

            moderate_threshold = float(
                THRESHOLDS.get(
                    "moderate",
                    0.33,
                )
            )

            high_threshold = float(
                THRESHOLDS.get(
                    "high",
                    0.66,
                )
            )


            if p < moderate_threshold:

                risk_tier = "Low"

            elif p < high_threshold:

                risk_tier = "Moderate"

            else:

                risk_tier = "High"


            # -----------------------------------------------
            # SHAP
            # -----------------------------------------------

            shap_output = EXPLAINER.shap_values(row)


            if isinstance(
                shap_output,
                list,
            ):

                if len(shap_output) > 1:

                    shap_values = shap_output[1]

                else:

                    shap_values = shap_output[0]

            else:

                shap_values = shap_output


            shap_values = np.asarray(
                shap_values
            )

            shap_values = np.squeeze(
                shap_values
            )


            # -----------------------------------------------
            # Handle SHAP dimensions
            # -----------------------------------------------

            if shap_values.ndim > 1:

                shap_values = shap_values.reshape(
                    -1
                )


            # XGBoost/SHAP versions can sometimes
            # return an extra output value.
            if len(shap_values) > len(FEATURES):

                shap_values = shap_values[
                    :len(FEATURES)
                ]


            # Safety fallback
            if len(shap_values) != len(FEATURES):

                raise ValueError(
                    "SHAP output does not match the "
                    "24 model features. "
                    f"Expected {len(FEATURES)}, "
                    f"received {len(shap_values)}."
                )


            # -----------------------------------------------
            # Explanation dataframe
            # -----------------------------------------------

            explanation = pd.DataFrame(
                {
                    "Feature": [
                        DISPLAY.get(
                            feature,
                            feature,
                        )
                        for feature in FEATURES
                    ],

                    "Feature Code": FEATURES,

                    "Patient Value": [
                        row.iloc[0][feature]
                        for feature in FEATURES
                    ],

                    "SHAP Value": shap_values,

                    "Absolute SHAP": np.abs(
                        shap_values
                    ),
                }
            )


            explanation = explanation.sort_values(
                "Absolute SHAP",
                ascending=False,
            ).reset_index(
                drop=True
            )


            explanation["Direction"] = np.where(
                explanation["SHAP Value"] >= 0,
                "Higher CKD model output",
                "Lower CKD model output",
            )


            # -----------------------------------------------
            # Save result
            # -----------------------------------------------

            st.session_state.last_result = {

                "input": row,

                "probability": p,

                "prediction": prediction,

                "risk_tier": risk_tier,

                "explanation": explanation,
            }


        except Exception as e:

            st.error(
                "An error occurred during prediction."
            )

            st.exception(e)


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    result = st.session_state.get(
        "last_result"
    )


    if result is not None:

        st.divider()

        st.markdown(
            "## 🩺 Assessment Result"
        )


        # -----------------------------------------------
        # Main metrics
        # -----------------------------------------------

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


        # -----------------------------------------------
        # Probability bar
        # -----------------------------------------------

        st.progress(
            result["probability"],
            text=(
                f"Predicted CKD probability: "
                f"{result['probability'] * 100:.2f}%"
            ),
        )


        # -----------------------------------------------
        # Risk explanation
        # -----------------------------------------------

        if result["risk_tier"] == "Low":

            st.info(
                "The model assigns this input to the "
                "Low research risk tier."
            )

        elif result["risk_tier"] == "Moderate":

            st.warning(
                "The model assigns this input to the "
                "Moderate research risk tier."
            )

        else:

            st.error(
                "The model assigns this input to the "
                "High research risk tier."
            )


        # -----------------------------------------------
        # SHAP explanation
        # -----------------------------------------------

        st.markdown(
            "### 🔎 Top Contributing Factors"
        )

        st.caption(
            "SHAP values explain the contribution of each "
            "feature to this individual model prediction. "
            "They do not represent clinical causation."
        )


        top_features = (
            result["explanation"]
            .head(8)
            .copy()
        )


        display_top = top_features[
            [
                "Feature",
                "Patient Value",
                "SHAP Value",
                "Direction",
            ]
        ]


        st.dataframe(
            display_top,
            use_container_width=True,
            hide_index=True,
        )


        # -----------------------------------------------
        # Download
        # -----------------------------------------------

        downloadable = {

            "prediction":
                result["prediction"],

            "ckd_probability":
                result["probability"],

            "risk_tier":
                result["risk_tier"],

            "model":
                "XGBoost",

            "explainability":
                "SHAP TreeExplainer",

            "patient_input":
                {
                    feature:
                    result["input"].iloc[0][feature]
                    for feature in FEATURES
                },

            "top_contributing_features":
                top_features[
                    [
                        "Feature",
                        "Feature Code",
                        "Patient Value",
                        "SHAP Value",
                        "Direction",
                    ]
                ].to_dict(
                    orient="records"
                ),

            "note":
                "Research/screening prototype; "
                "not a medical diagnosis.",
        }


        st.download_button(
            "⬇️ Download Assessment Result",
            data=json.dumps(
                downloadable,
                indent=2,
                default=str,
            ),
            file_name="ckd_assessment_result.json",
            mime="application/json",
        )


# ============================================================
# TAB 2 — EXPLAINABILITY
# ============================================================

with tab_explain:

    st.markdown(
        "## 🔎 Explainable AI"
    )


    st.write(
        "The application uses SHAP TreeExplainer to explain "
        "the XGBoost model prediction at feature level."
    )


    # ========================================================
    # GLOBAL SHAP
    # ========================================================

    st.markdown(
        "### Global Feature Importance"
    )


    if isinstance(
        SHAP_IMPORTANCE,
        list,
    ) and len(SHAP_IMPORTANCE) > 0:

        try:

            importance_df = pd.DataFrame(
                SHAP_IMPORTANCE
            )


            # Handle common artifact naming.
            feature_column = None
            value_column = None


            for candidate in [
                "feature",
                "Feature",
                "feature_name",
            ]:

                if candidate in importance_df.columns:

                    feature_column = candidate
                    break


            for candidate in [
                "mean_abs_shap",
                "Mean |SHAP|",
                "importance",
                "mean_abs",
            ]:

                if candidate in importance_df.columns:

                    value_column = candidate
                    break


            if (
                feature_column is not None
                and value_column is not None
            ):

                importance_df["Display Feature"] = (
                    importance_df[
                        feature_column
                    ].map(
                        lambda x:
                        DISPLAY.get(
                            x,
                            x,
                        )
                    )
                )


                importance_df[
                    "Importance"
                ] = pd.to_numeric(
                    importance_df[
                        value_column
                    ],
                    errors="coerce",
                )


                importance_df = (
                    importance_df
                    .sort_values(
                        "Importance",
                        ascending=False,
                    )
                )


                chart = (
                    importance_df
                    .head(12)
                    .set_index(
                        "Display Feature"
                    )[
                        "Importance"
                    ]
                )


                st.bar_chart(
                    chart
                )


                st.dataframe(
                    importance_df[
                        [
                            "Display Feature",
                            feature_column,
                            "Importance",
                        ]
                    ].rename(
                        columns={
                            "Display Feature":
                                "Feature",
                            feature_column:
                                "Feature Code",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


            else:

                st.info(
                    "The global SHAP artifact has an "
                    "unexpected structure."
                )

        except Exception as e:

            st.warning(
                "Unable to display global SHAP importance."
            )

            st.exception(e)

    else:

        st.info(
            "Global SHAP importance is not available."
        )


    # ========================================================
    # PATIENT SHAP
    # ========================================================

    result = st.session_state.get(
        "last_result"
    )


    if result is not None:

        st.markdown(
            "### Patient-Level SHAP Explanation"
        )


        patient_shap = (
            result["explanation"]
            .head(10)
            .copy()
            .sort_values(
                "SHAP Value"
            )
        )


        chart = (
            patient_shap
            .set_index(
                "Feature"
            )[
                "SHAP Value"
            ]
        )


        st.bar_chart(
            chart
        )


    else:

        st.info(
            "Complete a patient assessment first "
            "to view patient-level SHAP explanations."
        )


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================

with tab_performance:

    st.markdown(
        "## 📊 Model Performance"
    )


    st.caption(
        "Performance values correspond to the trained model "
        "artifact used by this application."
    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = METRICS.get(
        "accuracy"
    )

    precision = METRICS.get(
        "precision"
    )

    recall = METRICS.get(
        "recall"
    )

    f1 = METRICS.get(
        "f1"
    )

    auc = METRICS.get(
        "auc"
    )


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


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown(
        "### Confusion Matrix"
    )


    cm = METRICS.get(
        "confusion_matrix"
    )


    if cm is not None:

        try:

            cm_array = np.asarray(
                cm
            )


            cm_df = pd.DataFrame(
                cm_array,
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

    else:

        st.info(
            "Confusion matrix information is not available."
        )


    # ========================================================
    # DATASET
    # ========================================================

    st.markdown(
        "### Dataset Information"
    )


    d1, d2, d3 = st.columns(3)


    d1.metric(
        "Total Records",
        len(DATASET),
    )


    d2.metric(
        "Input Features",
        len(FEATURES),
    )


    if "class" in DATASET.columns:

        class_counts = (
            DATASET["class"]
            .value_counts()
            .to_dict()
        )


        d3.metric(
            "Class Categories",
            len(class_counts),
        )


# ============================================================
# TAB 4 — RESEARCH DETAILS
# ============================================================

with tab_research:

    st.markdown(
        "## 📚 Research Details"
    )


    st.markdown(
        """
        ### Research Title

        **Explainable AI-Based Framework for Early Detection
        and Risk Assessment of Chronic Kidney Disease Using
        Machine Learning**


        ### Framework

        **Clinical Inputs → Preprocessing → XGBoost →
        CKD Probability → Risk Assessment → SHAP Explanation**


        ### Input

        The model uses **24 clinical attributes** from the
        research dataset.


        ### Machine Learning Model

        **XGBoost** is used as the deployed prediction model.


        ### Explainability

        **SHAP TreeExplainer** is used to identify the
        contribution of individual features to the prediction.


        ### Risk Assessment

        The research prototype uses three probability tiers:

        - **Low:** probability < 0.33
        - **Moderate:** 0.33 ≤ probability < 0.66
        - **High:** probability ≥ 0.66


        ### Important

        These risk thresholds are research-design thresholds
        and are not clinically calibrated.
        """
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown(
        "### Deployment Model"
    )


    st.write(
        "Prediction model: **XGBoost**"
    )

    st.write(
        "Explainability method: **SHAP TreeExplainer**"
    )

    st.write(
        f"Number of input features: **{len(FEATURES)}**"
    )


    # ========================================================
    # SAFETY NOTE
    # ========================================================

    st.warning(
        "This application is a research prototype. "
        "A positive prediction does not establish a clinical "
        "diagnosis of chronic kidney disease."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CKD-AI Research Prototype | "
    "Explainable AI-Based CKD Detection and Risk Assessment"
)
