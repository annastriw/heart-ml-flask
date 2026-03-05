from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Tuple, Optional

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g
from flask_cors import CORS

# =========================================================
# CONFIG (override via env on production if needed)
# =========================================================
MODEL_PATH = os.getenv("MODEL_PATH", "saved_models/random_forest_tuned_best.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "saved_models/minmax_scaler.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "saved_models/feature_columns.json")

# Globals per-process (per waitress process / per gunicorn worker)
MODEL = None
SCALER = None
FEATURE_COLUMNS: List[str] = []

# Conservative lock for thread safety when using threaded servers
_PREDICT_LOCK = threading.Lock()


def load_artifacts() -> None:
    """Load model, scaler, and feature columns. Idempotent."""
    global MODEL, SCALER, FEATURE_COLUMNS

    if MODEL is not None and SCALER is not None and FEATURE_COLUMNS:
        return

    print("CWD:", os.getcwd())
    print("MODEL:", os.path.abspath(MODEL_PATH))
    print("SCALER:", os.path.abspath(SCALER_PATH))
    print("FEATURES:", os.path.abspath(FEATURES_PATH))

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    obj = joblib.load(MODEL_PATH)
    MODEL = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    SCALER = joblib.load(SCALER_PATH)

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"feature_columns.json not found: {FEATURES_PATH}")
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        cols = json.load(f)

    if not isinstance(cols, list) or not all(isinstance(x, str) for x in cols):
        raise ValueError("feature_columns.json must be a JSON array of strings")

    FEATURE_COLUMNS[:] = cols


# =========================================================
# INPUT FIELDS
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

MAPS = {
    "gender": {"male": 0, "female": 1},
    "smoking_status": {"never": 0, "past": 1, "current": 2},
    "physical_activity": {"low": 0, "moderate": 1, "high": 2},
    "dietary_habits": {"unhealthy": 0, "healthy": 1},
    "stress_level": {"low": 0, "moderate": 1, "high": 2},
}

RISK_RULES = {
    "age": (lambda x: x > 50, "Usia lanjut (>50 tahun)"),
    "blood_pressure_systolic": (lambda x: x > 140, "Tekanan darah sistolik tinggi (>140 mmHg)"),
    "blood_pressure_diastolic": (lambda x: x > 90, "Tekanan darah diastolik tinggi (>90 mmHg)"),
    "cholesterol_level": (lambda x: x > 240, "Kolesterol tinggi (>240 mg/dL)"),
    "bmi": (lambda x: x >= 25, "BMI tinggi (≥25)"),
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

FIELD_META = {
    "age": {"label": "Usia", "unit": "tahun"},
    "blood_pressure_systolic": {"label": "Tekanan darah sistolik", "unit": "mmHg"},
    "blood_pressure_diastolic": {"label": "Tekanan darah diastolik", "unit": "mmHg"},
    "hypertension": {"label": "Hipertensi", "unit": None},
    "random_blood_sugar": {"label": "Gula darah acak", "unit": "mg/dL"},
    "diabetes": {"label": "Diabetes", "unit": None},
    "cholesterol_level": {"label": "Kolesterol", "unit": "mg/dL"},
    "bmi": {"label": "BMI", "unit": None},
    "obesity": {"label": "Obesitas", "unit": None},
    "waist_circumference": {"label": "Lingkar pinggang", "unit": "cm"},
    "family_history": {"label": "Riwayat keluarga", "unit": None},
    "smoking_status": {"label": "Status merokok", "unit": None},
    "stress_level": {"label": "Tingkat stres", "unit": None},
    "sleep_hours": {"label": "Jam tidur", "unit": "jam"},
    "previous_heart_disease": {"label": "Riwayat penyakit jantung", "unit": None},
}

EXPLAIN_ORDER = [
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


# =========================================================
# Helpers
# =========================================================
def s(v: Any) -> str:
    return str(v).strip().lower()


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def to_bool(v: Any) -> bool:
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
    - {"record": {...}}
    - {...} (direct record)
    """
    if isinstance(payload.get("record"), dict):
        return payload["record"]
    if isinstance(payload, dict) and "personal_information_id" in payload and "blood_pressure_systolic" in payload:
        return payload
    return None


def ensure_required_meta(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    # rec["id"] == patient_health_check.id
    return rec.get("id"), rec.get("personal_information_id"), rec.get("name"), rec.get("check_date")


def normalize_backend_record(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      ml_row: encoded numeric row for model
      raw_norm: normalized values for explanation text
    """
    raw_norm: Dict[str, Any] = {}

    for f in NUM_FIELDS:
        raw_norm[f] = to_float(rec.get(f), 0.0)

    for f in BOOL_FIELDS:
        raw_norm[f] = to_bool(rec.get(f))

    for f in CAT_FIELDS:
        raw_norm[f] = s(rec.get(f, ""))

    ml_row: Dict[str, Any] = {}
    for f in NUM_FIELDS:
        ml_row[f] = raw_norm[f]
    for f in BOOL_FIELDS:
        ml_row[f] = int(raw_norm[f])
    for f in CAT_FIELDS:
        ml_row[f] = int(MAPS.get(f, {}).get(raw_norm[f], -1))

    return ml_row, raw_norm


def build_X(ml_row: Dict[str, Any]) -> pd.DataFrame:
    row = {col: ml_row.get(col, 0) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def top5_features_text() -> List[str]:
    """
    Returns list string:
      "1. cholesterol_level (0.081)"
    """
    if MODEL is None or not hasattr(MODEL, "feature_importances_"):
        return []
    imps = list(getattr(MODEL, "feature_importances_"))
    pairs = sorted(zip(FEATURE_COLUMNS, imps), key=lambda x: x[1], reverse=True)[:5]
    return [f"{i}. {name} ({float(score):.3f})" for i, (name, score) in enumerate(pairs, 1)]


def format_value_for_text(field: str, value: Any) -> str:
    if field in NUM_FIELDS:
        return f"{float(value):.1f}"
    if field in BOOL_FIELDS:
        return "yes" if bool(value) else "no"
    return str(value)


def build_analysis_text(raw_norm: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for f in EXPLAIN_ORDER:
        v = raw_norm.get(f)
        meta = FIELD_META.get(f, {"label": f, "unit": None})
        label = meta.get("label", f)
        unit = meta.get("unit")

        rule = RISK_RULES.get(f)
        is_risk = False
        message = "Normal"
        if rule is not None:
            cond, desc = rule
            if f in ("smoking_status", "stress_level"):
                is_risk = bool(cond(str(v)))
            else:
                is_risk = bool(cond(v))
            message = desc if is_risk else "Normal"

        val_txt = format_value_for_text(f, v)
        unit_txt = f" {unit}" if unit else ""
        if is_risk:
            out.append(f"{label}: {val_txt}{unit_txt} (risiko) → {message}")
        else:
            out.append(f"{label}: {val_txt}{unit_txt} (normal)")
    return out


# =========================================================
# APP FACTORY (required for waitress/gunicorn)
# =========================================================
def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Ensure artifacts are loaded when process starts
    load_artifacts()

    # -------------------------
    # Simple request logging
    # -------------------------
    @app.before_request
    def _start_timer():
        g._t0 = time.perf_counter()

    @app.after_request
    def _log_response(resp):
        try:
            ms = (time.perf_counter() - getattr(g, "_t0", time.perf_counter())) * 1000
            ip = request.headers.get("X-Forwarded-For", request.remote_addr)
            print(f"[{resp.status_code}] {request.method} {request.path} from {ip} - {ms:.1f}ms")
        except Exception:
            pass
        return resp

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
        if MODEL is None or SCALER is None or not FEATURE_COLUMNS:
            return jsonify({"status": "error", "message": "Artifacts not loaded"}), 500

        payload = request.get_json(silent=True) or {}
        rec = pick_record(payload)
        if rec is None:
            return jsonify({
                "status": "error",
                "message": "Invalid payload. Send {record:{...}} or a record object."
            }), 400

        health_check_id, personal_information_id, name, check_date = ensure_required_meta(rec)
        if not health_check_id or not personal_information_id or not name or not check_date:
            return jsonify({
                "status": "error",
                "message": "Missing required fields: id, personal_information_id, name, check_date"
            }), 422

        ml_row, raw_norm = normalize_backend_record(rec)
        X_df = build_X(ml_row)

        try:
            # Thread-safe critical section
            with _PREDICT_LOCK:
                X_scaled = SCALER.transform(X_df)

                # ✅ FIX sklearn warning:
                # model was trained with feature names; wrap scaled array back into DataFrame
                X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)

                pred = int(MODEL.predict(X_scaled_df)[0])
                if hasattr(MODEL, "predict_proba"):
                    proba = MODEL.predict_proba(X_scaled_df)[0]
                    not_risk = float(proba[0])
                    risk = float(proba[1])
                else:
                    risk = 1.0 if pred == 1 else 0.0
                    not_risk = 1.0 - risk
        except Exception as e:
            return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500

        return jsonify({
            "status": "success",
            "check": {
                "id": health_check_id,
                "personal_information_id": personal_information_id,
                "name": name,
                "check_date": check_date,
            },
            "pred": {
                "value": pred,
                "risk": risk,
                "not_risk": not_risk,
            },
            "text": {
                "factor": top5_features_text(),
                "analysis": build_analysis_text(raw_norm),
            }
        }), 200

    return app