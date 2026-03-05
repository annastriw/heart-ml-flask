from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "saved_models/random_forest_tuned_best.pkl"
SCALER_PATH = "saved_models/minmax_scaler.pkl"
FEATURES_PATH = "saved_models/feature_columns.json"

app = Flask(__name__)
CORS(app)

MODEL = None
SCALER = None
FEATURE_COLUMNS: List[str] = []


def load_artifacts() -> None:
    """Load model, scaler, and feature columns (required)."""
    global MODEL, SCALER, FEATURE_COLUMNS

    print("CWD:", os.getcwd())
    print("MODEL:", os.path.abspath(MODEL_PATH))
    print("SCALER:", os.path.abspath(SCALER_PATH))
    print("FEATURES:", os.path.abspath(FEATURES_PATH))

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    obj = joblib.load(MODEL_PATH)
    # your notebook saved dict {"model": rand_clf, ...}
    MODEL = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    SCALER = joblib.load(SCALER_PATH)

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"feature_columns.json not found: {FEATURES_PATH}")
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        FEATURE_COLUMNS[:] = json.load(f)

    if not isinstance(FEATURE_COLUMNS, list) or not all(isinstance(x, str) for x in FEATURE_COLUMNS):
        raise ValueError("feature_columns.json must be a JSON array of strings")


# =========================================================
# INPUT FORMAT (matches your backend record)
# =========================================================
NUM_FIELDS = [
    "age",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "random_blood_sugar",
    "cholesterol_level",
    "height",
    "weight",
    "bmi",
    "waist_circumference",
    "sleep_hours",
]

BOOL_FIELDS = [
    "hypertension",
    "diabetes",
    "obesity",
    "family_history",
    "previous_heart_disease",
    "medication_usage",
]

CAT_FIELDS = [
    "gender",
    "smoking_status",
    "physical_activity",
    "dietary_habits",
    "stress_level",
]

# =========================================================
# CATEGORICAL MAPPING (from your colab)
# note: backend sends "Male" / "NEVER" / etc -> we lowercase first
# =========================================================
MAPS = {
    "gender": {"male": 0, "female": 1},
    "smoking_status": {"never": 0, "past": 1, "current": 2},
    "physical_activity": {"low": 0, "moderate": 1, "high": 2},
    "dietary_habits": {"unhealthy": 0, "healthy": 1},
    "stress_level": {"low": 0, "moderate": 1, "high": 2},
}

# =========================================================
# RISK RULES (for explanation output)
# Use RAW values (before scaler).
# =========================================================
RISK_RULES = {
    "age": (lambda x: x > 50, "Usia lanjut (>50 tahun)"),
    "blood_pressure_systolic": (lambda x: x > 140, "Tekanan darah sistolik tinggi (>140 mmHg)"),
    "blood_pressure_diastolic": (lambda x: x > 90, "Tekanan darah diastolik tinggi (>90 mmHg)"),
    "cholesterol_level": (lambda x: x > 240, "Kolesterol tinggi (>240 mg/dL)"),
    "bmi": (lambda x: x >= 25, "Obesitas (BMI ≥25)"),
    "waist_circumference": (lambda x: x > 85, "Lingkar pinggang tinggi (>85 cm)"),
    "random_blood_sugar": (lambda x: x > 126, "Gula darah tinggi (>126 mg/dL)"),
    "sleep_hours": (lambda x: x < 6, "Kurang tidur (<6 jam)"),
    "hypertension": (lambda x: x is True, "Hipertensi"),
    "diabetes": (lambda x: x is True, "Diabetes"),
    "family_history": (lambda x: x is True, "Riwayat penyakit jantung keluarga"),
    "previous_heart_disease": (lambda x: x is True, "Riwayat penyakit jantung sebelumnya"),
    "obesity": (lambda x: x is True, "Obesitas"),
    "smoking_status": (lambda x: x == "current", "Perokok aktif"),
    "stress_level": (lambda x: x == "high", "Tingkat stres tinggi"),
}


# =========================================================
# Helpers
# =========================================================
def s(v: Any) -> str:
    return str(v).strip().lower()


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)  # works for "27.70" etc
    except Exception:
        return default


def to_bool(v: Any) -> bool:
    """
    Backend sends 0/1 (int) for boolean columns (as in your JSON).
    Accepts: bool, int, "0"/"1", "true"/"false".
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(int(v))
    vv = s(v)
    if vv in ("1", "true", "yes", "y", "iya", "ya"):
        return True
    if vv in ("0", "false", "no", "n", "tidak"):
        return False
    return False


def pick_record(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Accept either:
    - {"record": {...}}  (recommended)
    - {...}             (direct record)
    """
    if isinstance(payload.get("record"), dict):
        return payload["record"]
    if isinstance(payload, dict) and "personal_information_id" in payload and "blood_pressure_systolic" in payload:
        return payload
    return None


def normalize_backend_record(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      ml_row: dict numeric-coded for model input (before scaling)
      raw_norm: normalized raw values for explanation (float/bool/str)
    """
    raw_norm: Dict[str, Any] = {}

    # numeric
    for f in NUM_FIELDS:
        raw_norm[f] = to_float(rec.get(f), 0.0)

    # bool
    for f in BOOL_FIELDS:
        raw_norm[f] = to_bool(rec.get(f))

    # categorical (lowercase)
    for f in CAT_FIELDS:
        raw_norm[f] = s(rec.get(f, ""))

    # ml row
    ml_row: Dict[str, Any] = {}
    for f in NUM_FIELDS:
        ml_row[f] = raw_norm[f]

    for f in BOOL_FIELDS:
        ml_row[f] = int(raw_norm[f])

    for f in CAT_FIELDS:
        ml_row[f] = int(MAPS.get(f, {}).get(raw_norm[f], -1))  # unknown -> -1

    return ml_row, raw_norm


def build_X(ml_row: Dict[str, Any]) -> pd.DataFrame:
    """
    Build DataFrame with strict training feature columns order.
    If some fields missing -> default 0.
    """
    row = {col: ml_row.get(col, 0) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def explain_lines(raw_norm: Dict[str, Any]) -> List[str]:
    """
    EXACT style like your required output.
    """
    order = [
        "age",
        "blood_pressure_systolic",
        "blood_pressure_diastolic",
        "hypertension",
        "random_blood_sugar",
        "diabetes",
        "cholesterol_level",
        "bmi",
        "obesity",
        "waist_circumference",
        "family_history",
        "smoking_status",
        "stress_level",
        "sleep_hours",
        "previous_heart_disease",
    ]

    def label(f: str) -> str:
        return f.replace("_", " ").capitalize()

    lines: List[str] = []
    for f in order:
        v = raw_norm.get(f)

        if f in NUM_FIELDS:
            val_txt = f"{float(v):.1f}"
        elif f in BOOL_FIELDS:
            val_txt = "yes" if bool(v) else "no"
        else:
            val_txt = str(v)

        rule = RISK_RULES.get(f)
        if rule is None:
            lines.append(f"  ✓ {label(f)}: {val_txt} (normal)")
            continue

        cond, desc = rule
        if f in ("smoking_status", "stress_level"):
            risky = cond(str(v))
        else:
            risky = cond(v)

        if risky:
            lines.append(f"  🔸 {label(f)}: {val_txt} → {desc}")
        else:
            lines.append(f"  ✓ {label(f)}: {val_txt} (normal)")

    return lines


def top5_features() -> List[Tuple[str, float]]:
    if MODEL is None or not hasattr(MODEL, "feature_importances_"):
        return []
    imps = list(getattr(MODEL, "feature_importances_"))
    pairs = sorted(zip(FEATURE_COLUMNS, imps), key=lambda x: x[1], reverse=True)[:5]
    return [(name, float(score)) for name, score in pairs]


def format_text(prob_risk: float, prob_not: float, pred: int, risk_lines: List[str], top5: List[Tuple[str, float]]) -> str:
    """
    REQUIRED output format in a single text block.
    """
    out: List[str] = []
    out.append(f"Probabilitas → Risiko: {prob_risk*100:.2f}% | Tidak Risiko: {prob_not*100:.2f}%")
    out.append("Hasil: BERISIKO TINGGI TERKENA PENYAKIT JANTUNG" if pred == 1 else "Hasil: RISIKO RENDAH / NORMAL")
    out.append("Analisis faktor risiko:")
    out.extend(risk_lines)
    out.append("Top 5 fitur paling berpengaruh:")
    if top5:
        for i, (name, score) in enumerate(top5, 1):
            out.append(f"  {i}. {name} (importance: {score:.3f})")
    else:
        # keep 5 lines
        for i in range(1, 6):
            out.append(f"  {i}. - (importance: 0.000)")
    return "\n".join(out)


# =========================================================
# ROUTES
# =========================================================
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "artifacts": {
            "model_loaded": MODEL is not None,
            "scaler_loaded": SCALER is not None,
            "feature_columns_loaded": bool(FEATURE_COLUMNS),
            "feature_count": len(FEATURE_COLUMNS),
        }
    }), 200


@app.post("/predict-heart-attack")
def predict_heart_attack():
    """
    Accept payload:
      1) {"record": <backend-record>}
      2) <backend-record> directly

    backend-record example (same as your selected item):
      {
        "id": "...",
        "personal_information_id": "...",
        "age": "25",
        "gender": "Male",
        "hypertension": 0,
        ...
      }
    """
    if MODEL is None or SCALER is None or not FEATURE_COLUMNS:
        return jsonify({"status": "error", "message": "Artifacts not loaded"}), 500

    payload = request.get_json(silent=True) or {}
    rec = pick_record(payload)
    if rec is None:
        return jsonify({
            "status": "error",
            "message": "Invalid payload. Send {record:{...}} or a record object."
        }), 400

    # normalize + encode
    ml_row, raw_norm = normalize_backend_record(rec)

    # build feature frame
    X_df = build_X(ml_row)

    # scale (scaler fitted with feature names -> passing df keeps it consistent)
    try:
        X_scaled = SCALER.transform(X_df)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Scaling failed: {str(e)}"}), 500

    # predict
    try:
        pred = int(MODEL.predict(X_scaled)[0])
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X_scaled)[0]
            prob_not = float(proba[0])
            prob_risk = float(proba[1])
        else:
            prob_risk = 1.0 if pred == 1 else 0.0
            prob_not = 1.0 - prob_risk
    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500

    # explanation + top5
    risk_lines = explain_lines(raw_norm)
    top5 = top5_features()

    # required formatted text
    text = format_text(prob_risk, prob_not, pred, risk_lines, top5)

    return jsonify({
        "status": "success",
        "text": text,  # <- this is the exact format block you want
        "data": {
            "pred": pred,
            "prob_risk": prob_risk,
            "prob_not_risk": prob_not,
            "top5": [{"feature": n, "importance": s} for n, s in top5],
            "record_meta": {
                "id": rec.get("id"),
                "personal_information_id": rec.get("personal_information_id"),
                "name": rec.get("name"),
                "check_date": rec.get("check_date"),
            }
        }
    }), 200


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=True)