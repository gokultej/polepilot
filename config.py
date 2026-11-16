"""
config.py - runtime settings for raw prediction mode.
"""

import os

# -- Detection thresholds -----------------------------------------------------
DETECTION_CONF = 0.20
DETECTION_CONF_POLE = 0.15  # Candidate collection; acceptance uses THRESHOLD_POLE.
DETECTION_IOU = 0.45

THRESHOLD_INSULATOR = 0.20
THRESHOLD_CROSSARM = 0.20
THRESHOLD_POLE = 0.80
THRESHOLD_CONDUCTOR = 0.01

POLE_STAGE_IMGSZ = 640
COMPONENT_STAGE_IMGSZ = 1024
INSULATOR_STAGE_IMGSZ = 1024
SHED_STAGE_IMGSZ = 1024
CONDUCTOR_STAGE_IMGSZ = 1024

# -- Minimal geometry settings still used by the UI/pipeline ------------------
POLE_IDEAL_ANGLE_DEG = 90.0

# -- Raw class keyword routing ------------------------------------------------
OBB_CLASS_KEYWORDS = {
    "insulator": ["insulator", "ht_pin", "ht_disc", "lt_pin", "shackle"],
    "pole": ["pole", "ht pole", "lt pole", "ht/lt pole", "support"],
    "crossarm": [
        "crossarm", "arm", "bracket", "t_rising", "t_raising",
        "v_cross", "channel", "cleat", "tapping", "vcb",
        "3_wire", "5_wire", "3_wire_crossarm", "5_wire_crossarm", "3 wire cross arm", "5 wire cross arm"
    ],
    "stay_set": ["stay_set", "stay set", "stay", "guy_wire"],
    "conductor": ["conductor", "wire", "line", "street light wire", "cable"],
    "dtr_tank": ["transformer", "dtr", "25 kva dtr ht pole", "63 kva dtr ht pole", "100 kva dtr ht pole"],
    "lamp_head": ["lamp", "street_light", "street light"],
    "broken_wire": ["wire_broken", "broken_wire", "snapped"],
    "vegetation": ["vegetation", "tree", "plant", "bush"],
}


def load_dotenv(dotenv_path=".env"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, dotenv_path)
    if not os.path.exists(full_path):
        return

    with open(full_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip(chr(39) + chr(34)))


load_dotenv()

APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "dev-secret-change-me")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "")
PG_DB = os.getenv("PG_DB", "gis_yolo")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH = os.getenv("SQLITE_PATH", os.path.join(BASE_DIR, "gis_yolo.db"))

# -- Pole Auto-Survey Redirection & Auto-Fill Toggle ---------------------------
ENABLE_POLE_AUTO_SURVEY_REDIRECT = os.getenv("ENABLE_POLE_AUTO_SURVEY_REDIRECT", "true").lower() in ("true", "1", "yes")


def normalize_detection_label(label, fallback="object"):
    cleaned = str(label or fallback).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or fallback


def is_pole_label(label):
    return normalize_detection_label(label) in {
        'pole', 'main_pole', 'strut_pole', 'stut_pole', 'vertical_pole',
        'ht_pole', 'lt_pole', 'ht/lt_pole', 'support',
    }


def meets_pole_confidence(confidence):
    import math
    if isinstance(confidence, bool):
        return False
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return False
    return math.isfinite(score) and THRESHOLD_POLE <= score <= 1.0


def filter_pole_detections(detections):
    """Preserve other classes; reject unscored or sub-threshold pole labels."""
    return [d for d in detections if not is_pole_label(d.get('label')) or meets_pole_confidence(d.get('confidence'))]
