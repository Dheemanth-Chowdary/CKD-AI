"""Regression test for the CKD-AI three-tier demo-profile generator.

Extracts the exact find_demo_profiles() function from app.py and executes it
against the committed model artifact and dataset. It fails if the model cannot
produce a score inside each configured risk band.
"""
import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
MODEL_PATH = ROOT / "artifacts" / "ckd_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"
DATASET_PATH = ROOT / "ckd_dataset.csv"

FEATURES_DEFAULT = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane",
]
BINARY_FEATURES = {
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"
}
ORDINAL_FEATURES = {"al", "su"}


def main():
    with MODEL_PATH.open("rb") as file:
        bundle = joblib.load(file)

    model = bundle["model"]
    features = list(bundle.get("features", FEATURES_DEFAULT))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    thresholds = bundle.get(
        "risk_thresholds",
        metadata.get("risk_thresholds", {"moderate": 0.33, "high": 0.66}),
    )
    dataset = pd.read_csv(DATASET_PATH)

    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    function_node = next(
        (
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "find_demo_profiles"
        ),
        None,
    )
    if function_node is None:
        raise AssertionError("find_demo_profiles() was not found in app.py")

    namespace = {
        "np": np,
        "pd": pd,
        "DATASET": dataset,
        "FEATURES": features,
        "RISK_THRESHOLDS": thresholds,
        "MODEL": model,
        "BINARY_FEATURES": BINARY_FEATURES,
        "ORDINAL_FEATURES": ORDINAL_FEATURES,
    }
    extracted_module = ast.Module(body=[function_node], type_ignores=[])
    exec(compile(extracted_module, str(APP_PATH), "exec"), namespace)
    profiles = namespace["find_demo_profiles"]()

    moderate = float(thresholds.get("moderate", 0.33))
    high = float(thresholds.get("high", 0.66))
    checks = {
        "Low": lambda p: p < moderate,
        "Moderate": lambda p: moderate <= p < high,
        "High": lambda p: p >= high,
    }

    missing = [tier for tier in checks if tier not in profiles]
    if missing:
        raise AssertionError(
            "Model did not yield examples for: " + ", ".join(missing)
        )

    print("Three-tier demo profile verification:")
    print(f"Thresholds: Low < {moderate:.3f}; Moderate [{moderate:.3f}, {high:.3f}); High >= {high:.3f}")
    for tier in ("High", "Moderate", "Low"):
        profile = profiles[tier]
        probability = float(profile["probability"])
        recheck = float(model.predict_proba(profile["input"][features])[0, 1])
        if not checks[tier](probability):
            raise AssertionError(f"{tier} profile has out-of-band probability {probability:.6f}")
        if abs(recheck - probability) > 1e-7:
            raise AssertionError(
                f"{tier} profile probability changed: saved={probability:.8f}, model={recheck:.8f}"
            )
        print(f"PASS {tier:8s} probability={probability:.6f} ({probability * 100:.2f}%)")

    print("PASS: all three risk levels are backed by actual model scores.")


if __name__ == "__main__":
    main()
