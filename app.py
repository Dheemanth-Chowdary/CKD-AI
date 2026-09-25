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
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "ckd_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
SHAP_PATH = ARTIFACT_DIR / "shap_importance.json"
DATASET_PATH = BASE_DIR / "ckd_dataset.csv"


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


DISPLAY_NAMES = {
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


# Binary features in the supplied dataset.
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


# Ordinal categorical numerical features.
ORDINAL_FEATURES = {
    "al",
    "su",
}


# ============================================================
# BASIC CSS
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
        background-color: #f4f6f8;
        border: 1px solid #d7dce2;
        font-size: 0.92rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = {
    "Model": MODEL_PATH,
    "Metadata": METADATA_PATH,
    "Metrics": METRICS_PATH,
    "SHAP importance": SHAP_PATH,
    "Dataset": DATASET_PATH,
}


missing_files = []

for file_name, file_path in required_files.items():

    if not file_path.exists():

        missing_files.append(
            f"{file_name}: {file_path}"
        )


if missing_files:

    st.error(
        "Required project files are missing."
    )

    for item in missing_files:

        st.write(
            f"- `{item}`"
        )

    st.stop()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:

    bundle = joblib.load(
        MODEL_PATH
    )

    MODEL = bundle["model"]

    MODEL_FEATURES = bundle.get(
        "features",
        FEATURES,
    )

    RISK_THRESHOLDS = bundle.get(
        "risk_thresholds",
        {
            "moderate": 0.33,
            "high": 0.66,
        },
    )

except Exception as e:

    st.error(
        "Unable to load the trained CKD model."
    )

    st.exception(e)

    st.stop()


# Use the exact feature order stored with the model.
if MODEL_FEATURES:

    FEATURES = list(
        MODEL_FEATURES
    )


# ============================================================
# LOAD METADATA
# ============================================================

try:

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        METADATA = json.load(f)

except Exception as e:

    st.warning(
        "metadata.json could not be loaded. "
        "The application will continue with limited metadata."
    )

    METADATA = {}


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
# LOAD SHAP IMPORTANCE
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

    DATASET = pd.read_csv(
        DATASET_PATH
    )

except Exception as e:

    st.error(
        "Unable to read ckd_dataset.csv."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

try:

    # Do NOT cache this function with the XGBoost model as an
    # argument. Streamlit may attempt to hash XGBClassifier.
    EXPLAINER = shap.TreeExplainer(
        MODEL
    )

except Exception as e:

    st.error(
        "Unable to initialize the SHAP explainer."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "last_result" not in st.session_state:

    st.session_state.last_result = None


# ============================================================
# SESSION STATE COMPATIBILITY
# ============================================================

# Previous versions of the application used:
#
#     label
#     tier
#
# The current application uses:
#
#     prediction
#     risk_tier
#
# This block prevents old Streamlit sessions from crashing.

old_result = st.session_state.get(
    "last_result"
)


if old_result is not None:

    if (
        "prediction" not in old_result
        and "label" in old_result
    ):

        old_result["prediction"] = (
            old_result["label"]
        )


    if (
        "risk_tier" not in old_result
        and "tier" in old_result
    ):

        old_result["risk_tier"] = (
            old_result["tier"]
        )


    required_result_keys = {
        "prediction",
        "risk_tier",
        "probability",
        "explanation",
        "input",
    }


    if not required_result_keys.issubset(
        old_result.keys()
    ):

        st.session_state.last_result = None


# ============================================================
# HEADER
# ============================================================

st.title(
    "🩺 CKD-AI"
)

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

    <b>Research / Screening Prototype</b><br><br>

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

tab_assessment, tab_explainability, tab_performance, tab_research = st.tabs(
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

    st.markdown(
        "## 🧪 Patient Clinical Assessment"
    )

    st.info(
        "Enter all 24 clinical attributes. "
        "The fields are intentionally blank when the application "
        "starts so that the model does not make a prediction from "
        "automatically inserted median values."
    )


    # ========================================================
    # SECTION 1 — BASIC MEASUREMENTS
    # ========================================================

    st.markdown(
        "### 1. Basic Measurements"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=120.0,
            value=None,
            step=1.0,
            placeholder="Enter age",
        )


    with col2:

        bp = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=300.0,
            value=None,
            step=1.0,
            placeholder="Enter blood pressure",
        )


    with col3:

        sg = st.number_input(
            "Specific Gravity",
            min_value=0.0,
            max_value=2.0,
            value=None,
            step=0.001,
            format="%.3f",
            placeholder="Example: 1.020",
        )


    # ========================================================
    # SECTION 2 — URINE PARAMETERS
    # ========================================================

    st.markdown(
        "### 2. Urine Parameters"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        al = st.selectbox(
            "Albumin",
            options=[0, 1, 2, 3, 4, 5],
            index=None,
            placeholder="Select Albumin",
            help=(
                "Ordinal numerical encoding used in the "
                "supplied research dataset."
            ),
        )


    with col2:

        su = st.selectbox(
            "Sugar",
            options=[0, 1, 2, 3, 4, 5],
            index=None,
            placeholder="Select Sugar",
            help=(
                "Ordinal numerical encoding used in the "
                "supplied research dataset."
            ),
        )


    with col3:

        rbc = st.selectbox(
            "Red Blood Cells",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        pc = st.selectbox(
            "Pus Cell",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    with col2:

        pcc = st.selectbox(
            "Pus Cell Clumps",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    with col3:

        ba = st.selectbox(
            "Bacteria",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    # ========================================================
    # SECTION 3 — BLOOD PARAMETERS
    # ========================================================

    st.markdown(
        "### 3. Blood Parameters"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        bgr = st.number_input(
            "Blood Glucose Random",
            min_value=0.0,
            max_value=1000.0,
            value=None,
            step=1.0,
            placeholder="Enter value",
        )


    with col2:

        bu = st.number_input(
            "Blood Urea",
            min_value=0.0,
            max_value=500.0,
            value=None,
            step=1.0,
            placeholder="Enter value",
        )


    with col3:

        sc = st.number_input(
            "Serum Creatinine",
            min_value=0.0,
            max_value=100.0,
            value=None,
            step=0.1,
            placeholder="Enter value",
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        sod = st.number_input(
            "Sodium",
            min_value=0.0,
            max_value=250.0,
            value=None,
            step=0.1,
            placeholder="Enter value",
        )


    with col2:

        pot = st.number_input(
            "Potassium",
            min_value=0.0,
            max_value=20.0,
            value=None,
            step=0.1,
            placeholder="Enter value",
        )


    with col3:

        hemo = st.number_input(
            "Haemoglobin",
            min_value=0.0,
            max_value=30.0,
            value=None,
            step=0.1,
            placeholder="Enter value",
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        pcv = st.number_input(
            "Packed Cell Volume",
            min_value=0.0,
            max_value=100.0,
            value=None,
            step=0.1,
            placeholder="Enter value",
        )


    with col2:

        wc = st.number_input(
            "White Blood Cell Count",
            min_value=0.0,
            max_value=50000.0,
            value=None,
            step=10.0,
            placeholder="Enter value",
        )


    with col3:

        rc = st.number_input(
            "Red Blood Cell Count",
            min_value=0.0,
            max_value=20.0,
            value=None,
            step=0.1,
            placeholder="Enter value",
        )


    # ========================================================
    # SECTION 4 — MEDICAL CONDITIONS
    # ========================================================

    st.markdown(
        "### 4. Medical Condition Indicators"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        htn = st.selectbox(
            "Hypertension",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    with col2:

        dm = st.selectbox(
            "Diabetes Mellitus",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    with col3:

        cad = st.selectbox(
            "Coronary Artery Disease",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        appet = st.selectbox(
            "Appetite",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    with col2:

        pe = st.selectbox(
            "Pedal Edema",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    with col3:

        ane = st.selectbox(
            "Anaemia",
            options=[0, 1],
            index=None,
            placeholder="Select 0 or 1",
        )


    # ========================================================
    # COLLECT ALL INPUTS
    # ========================================================

    patient_values = {

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

    missing_fields = []

    for feature in FEATURES:

        if patient_values.get(feature) is None:

            missing_fields.append(
                DISPLAY_NAMES.get(
                    feature,
                    feature,
                )
            )


    if missing_fields:

        st.warning(
            f"{len(missing_fields)} clinical field(s) "
            "still need to be entered."
        )

        with st.expander(
            "View missing fields"
        ):

            for field in missing_fields:

                st.write(
                    f"• {field}"
                )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    st.divider()


    analyze_button = st.button(
        "🔍 Analyze CKD Risk",
        type="primary",
        use_container_width=True,
        disabled=len(missing_fields) > 0,
    )


    # ========================================================
    # RUN MODEL
    # ========================================================

    if analyze_button:

        try:

            # ------------------------------------------------
            # Prepare model input
            # ------------------------------------------------

            model_values = []


            for feature in FEATURES:

                value = patient_values[
                    feature
                ]


                if feature in BINARY_FEATURES:

                    value = int(value)


                elif feature in ORDINAL_FEATURES:

                    value = int(value)


                else:

                    value = float(value)


                model_values.append(
                    value
                )


            input_df = pd.DataFrame(
                [model_values],
                columns=FEATURES,
            )


            # ------------------------------------------------
            # DATASET RANGE CHECK
            # ------------------------------------------------

            dataset_ranges = METADATA.get(
                "dataset_ranges",
                {},
            )


            outside_range = []


            for feature in FEATURES:

                if feature not in dataset_ranges:

                    continue


                try:

                    minimum = float(
                        dataset_ranges[
                            feature
                        ]["min"]
                    )

                    maximum = float(
                        dataset_ranges[
                            feature
                        ]["max"]
                    )

                    current_value = float(
                        input_df.iloc[0][
                            feature
                        ]
                    )


                    if (
                        current_value < minimum
                        or current_value > maximum
                    ):

                        outside_range.append(
                            {
                                "feature":
                                    DISPLAY_NAMES.get(
                                        feature,
                                        feature,
                                    ),

                                "value":
                                    current_value,

                                "minimum":
                                    minimum,

                                "maximum":
                                    maximum,
                            }
                        )

                except Exception:

                    continue


            if outside_range:

                st.warning(
                    "Some entered values are outside the "
                    "empirical range of the supplied training dataset. "
                    "The model will receive these values without clipping."
                )


                with st.expander(
                    "Values outside training-data range"
                ):

                    for item in outside_range:

                        st.write(
                            f"• {item['feature']}: "
                            f"{item['value']} "
                            f"(dataset range "
                            f"{item['minimum']}–"
                            f"{item['maximum']})"
                        )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            probabilities = MODEL.predict_proba(
                input_df
            )


            probability = float(
                probabilities[0][1]
            )


            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

            prediction = (
                "CKD Positive"
                if probability >= 0.50
                else "Not CKD"
            )


            # ------------------------------------------------
            # RISK THRESHOLDS
            # ------------------------------------------------

            moderate_threshold = float(
                RISK_THRESHOLDS.get(
                    "moderate",
                    0.33,
                )
            )


            high_threshold = float(
                RISK_THRESHOLDS.get(
                    "high",
                    0.66,
                )
            )


            if probability < moderate_threshold:

                risk_tier = "Low"


            elif probability < high_threshold:

                risk_tier = "Moderate"


            else:

                risk_tier = "High"


            # ------------------------------------------------
            # SHAP EXPLANATION
            # ------------------------------------------------

            shap_output = EXPLAINER.shap_values(
                input_df
            )


            # Handle SHAP binary classification
            # outputs across versions.

            if isinstance(
                shap_output,
                list,
            ):

                if len(shap_output) >= 2:

                    shap_values = (
                        shap_output[1]
                    )

                else:

                    shap_values = (
                        shap_output[0]
                    )

            else:

                shap_values = shap_output


            shap_values = np.asarray(
                shap_values
            )


            shap_values = np.squeeze(
                shap_values
            )


            if shap_values.ndim > 1:

                shap_values = (
                    shap_values.reshape(-1)
                )


            # Some SHAP versions may return
            # an additional value.

            if len(shap_values) > len(
                FEATURES
            ):

                shap_values = (
                    shap_values[
                        :len(FEATURES)
                    ]
                )


            if len(shap_values) != len(
                FEATURES
            ):

                raise ValueError(
                    "SHAP output does not match "
                    "the model feature count. "
                    f"Expected {len(FEATURES)}, "
                    f"received {len(shap_values)}."
                )


            # ------------------------------------------------
            # EXPLANATION TABLE
            # ------------------------------------------------

            explanation = pd.DataFrame(
                {
                    "Feature": [
                        DISPLAY_NAMES.get(
                            feature,
                            feature,
                        )
                        for feature in FEATURES
                    ],

                    "Feature Code": FEATURES,

                    "Patient Value": [
                        input_df.iloc[0][
                            feature
                        ]
                        for feature in FEATURES
                    ],

                    "SHAP Value": shap_values,

                    "Absolute SHAP": np.abs(
                        shap_values
                    ),
                }
            )


            explanation = (
                explanation
                .sort_values(
                    "Absolute SHAP",
                    ascending=False,
                )
                .reset_index(
                    drop=True
                )
            )


            explanation[
                "Direction"
            ] = np.where(
                explanation[
                    "SHAP Value"
                ] >= 0,

                "Increases CKD model output",

                "Decreases CKD model output",
            )


            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            st.session_state.last_result = {

                "input":
                    input_df,

                "probability":
                    probability,

                "prediction":
                    prediction,

                "risk_tier":
                    risk_tier,

                "explanation":
                    explanation,
            }


        except Exception as e:

            st.error(
                "An error occurred while analyzing "
                "the patient data."
            )

            st.exception(e)


    # ========================================================
    # DISPLAY LAST RESULT
    # ========================================================

    result = st.session_state.get(
        "last_result"
    )


    # Compatibility with previous versions.
    if result is not None:

        if (
            "prediction" not in result
            and "label" in result
        ):

            result["prediction"] = (
                result["label"]
            )


        if (
            "risk_tier" not in result
            and "tier" in result
        ):

            result["risk_tier"] = (
                result["tier"]
            )


    if result is not None:

        required_keys = {
            "input",
            "probability",
            "prediction",
            "risk_tier",
            "explanation",
        }


        if not required_keys.issubset(
            result.keys()
        ):

            st.session_state.last_result = None

            result = None


    # ========================================================
    # RESULT DISPLAY
    # ========================================================

    if result is not None:

        st.divider()

        st.markdown(
            "## 🩺 Assessment Result"
        )


        # ----------------------------------------------------
        # RESULT METRICS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Prediction",
                result[
                    "prediction"
                ],
            )


        with col2:

            st.metric(
                "CKD Probability",
                (
                    f"{result['probability'] * 100:.2f}%"
                ),
            )


        with col3:

            st.metric(
                "Research Risk Tier",
                result[
                    "risk_tier"
                ],
            )


        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------

        st.progress(
            min(
                max(
                    result[
                        "probability"
                    ],
                    0.0,
                ),
                1.0,
            ),
            text=(
                "Predicted CKD probability: "
                f"{result['probability'] * 100:.2f}%"
            ),
        )


        # ----------------------------------------------------
        # RISK MESSAGE
        # ----------------------------------------------------

        if result["risk_tier"] == "Low":

            st.info(
                "The model assigns this patient input "
                "to the Low research risk tier."
            )


        elif result["risk_tier"] == "Moderate":

            st.warning(
                "The model assigns this patient input "
                "to the Moderate research risk tier."
            )


        else:

            st.error(
                "The model assigns this patient input "
                "to the High research risk tier."
            )


        # ----------------------------------------------------
        # TOP FEATURES
        # ----------------------------------------------------

        st.markdown(
            "### 🔎 Top Contributing Factors"
        )


        st.caption(
            "SHAP values describe how individual features "
            "contribute to this model prediction. They do "
            "not represent clinical causation."
        )


        top_features = (
            result[
                "explanation"
            ]
            .head(8)
            .copy()
        )


        st.dataframe(
            top_features[
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


        # ----------------------------------------------------
        # DOWNLOAD RESULT
        # ----------------------------------------------------

        downloadable_result = {

            "prediction":
                result[
                    "prediction"
                ],

            "ckd_probability":
                result[
                    "probability"
                ],

            "risk_tier":
                result[
                    "risk_tier"
                ],

            "model":
                "XGBoost",

            "explainability":
                "SHAP TreeExplainer",

            "patient_input":
                {
                    feature:
                    result[
                        "input"
                    ].iloc[0][feature]

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
                downloadable_result,
                indent=2,
                default=str,
            ),
            file_name=(
                "ckd_assessment_result.json"
            ),
            mime="application/json",
        )


# ============================================================
# TAB 2 — EXPLAINABILITY
# ============================================================

with tab_explainability:

    st.markdown(
        "## 🔎 Explainable AI"
    )


    st.write(
        "SHAP TreeExplainer is used to explain the "
        "XGBoost model at both global and patient levels."
    )


    # ========================================================
    # GLOBAL SHAP
    # ========================================================

    st.markdown(
        "### Global Feature Importance"
    )


    try:

        if isinstance(
            SHAP_IMPORTANCE,
            list,
        ) and len(
            SHAP_IMPORTANCE
        ) > 0:

            importance_df = pd.DataFrame(
                SHAP_IMPORTANCE
            )


            feature_column = None
            importance_column = None


            for candidate in [
                "feature",
                "Feature",
                "feature_name",
            ]:

                if candidate in (
                    importance_df.columns
                ):

                    feature_column = candidate
                    break


            for candidate in [
                "mean_abs_shap",
                "Mean |SHAP|",
                "importance",
                "mean_abs",
            ]:

                if candidate in (
                    importance_df.columns
                ):

                    importance_column = candidate
                    break


            if (
                feature_column
                and importance_column
            ):

                importance_df[
                    "Display Feature"
                ] = (
                    importance_df[
                        feature_column
                    ].map(
                        lambda x:
                        DISPLAY_NAMES.get(
                            x,
                            x,
                        )
                    )
                )


                importance_df[
                    "Importance"
                ] = pd.to_numeric(
                    importance_df[
                        importance_column
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


                chart_data = (
                    importance_df
                    .head(12)
                    .set_index(
                        "Display Feature"
                    )[
                        "Importance"
                    ]
                )


                st.bar_chart(
                    chart_data
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
                    "The global SHAP artifact does not "
                    "contain the expected feature/importance columns."
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


    # ========================================================
    # PATIENT-LEVEL SHAP
    # ========================================================

    result = st.session_state.get(
        "last_result"
    )


    if result is not None:

        st.markdown(
            "### Patient-Level SHAP Explanation"
        )


        patient_shap = (
            result[
                "explanation"
            ]
            .head(10)
            .copy()
            .sort_values(
                "SHAP Value"
            )
        )


        chart_data = (
            patient_shap
            .set_index(
                "Feature"
            )[
                "SHAP Value"
            ]
        )


        st.bar_chart(
            chart_data
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
        "These values correspond to the trained model "
        "artifact used by this application."
    )


    # ========================================================
    # PERFORMANCE METRICS
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


    col1, col2, col3, col4, col5 = st.columns(5)


    if accuracy is not None:

        col1.metric(
            "Accuracy",
            f"{float(accuracy) * 100:.2f}%",
        )


    if precision is not None:

        col2.metric(
            "Precision",
            f"{float(precision) * 100:.2f}%",
        )


    if recall is not None:

        col3.metric(
            "Recall",
            f"{float(recall) * 100:.2f}%",
        )


    if f1 is not None:

        col4.metric(
            "F1 Score",
            f"{float(f1) * 100:.2f}%",
        )


    if auc is not None:

        col5.metric(
            "ROC-AUC",
            f"{float(auc):.3f}",
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown(
        "### Confusion Matrix"
    )


    confusion_matrix = METRICS.get(
        "confusion_matrix"
    )


    if confusion_matrix is not None:

        try:

            cm = np.asarray(
                confusion_matrix
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


        except Exception:

            st.write(
                confusion_matrix
            )


    else:

        st.info(
            "Confusion matrix information is not available."
        )


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.markdown(
        "### Dataset Information"
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Total Records",
        len(DATASET),
    )


    col2.metric(
        "Input Features",
        len(FEATURES),
    )


    if "class" in DATASET.columns:

        class_count = (
            DATASET["class"]
            .nunique()
        )

        col3.metric(
            "Class Categories",
            class_count,
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


        ### Proposed Framework

        **Clinical Data → Preprocessing → Machine Learning →
        CKD Probability → Risk Assessment → SHAP Explanation**


        ### Input

        The deployed model uses **24 clinical attributes**.


        ### Machine Learning Model

        **XGBoost** is used as the deployed prediction model.


        ### Explainability

        **SHAP TreeExplainer** is used to identify the
        contribution of individual features to the model output.


        ### Research Risk Tiers

        **Low:** probability < 0.33

        **Moderate:** 0.33 ≤ probability < 0.66

        **High:** probability ≥ 0.66


        ### Important Note

        These probability thresholds are research-design
        thresholds and are not clinically calibrated.
        """
    )


    # ========================================================
    # MODEL DETAILS
    # ========================================================

    st.markdown(
        "### Deployment Information"
    )


    st.write(
        "Prediction model: **XGBoost**"
    )


    st.write(
        "Explainability method: **SHAP TreeExplainer**"
    )


    st.write(
        f"Input features: **{len(FEATURES)}**"
    )


    # ========================================================
    # SAFETY
    # ========================================================

    st.warning(
        "A CKD-positive model prediction does not establish "
        "a clinical diagnosis. The application is intended "
        "for research and screening demonstration."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CKD-AI Research Prototype | "
    "Explainable AI-Based CKD Detection and Risk Assessment"
)
