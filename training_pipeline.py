"""
training_pipeline.py
====================
Automated Active Learning Pipeline for Infrastructure Detection.

When an asset is approved by an admin:
  1. Images + corrected annotations are exported to YOLO format
  2. Training sample count is tracked in the DB
  3. When threshold is reached, a retrain job is queued

Directory structure created on server:
  training_data/
    images/         ← JPEG images
    labels/         ← YOLO .txt label files
    data.yaml       ← YOLO dataset config
    log.json        ← Training history log
"""

import os
import ast
import json
import base64
import uuid
from db import get_db_connection, init_postgres_db, parse_db_json
from datetime import datetime
import pathlib
from collections import Counter, defaultdict
import time

# ─── Configuration ────────────────────────────────────────────────────────────
TRAINING_DIR      = "training_data"
IMAGES_DIR        = os.path.join(TRAINING_DIR, "images")
LABELS_DIR        = os.path.join(TRAINING_DIR, "labels")
LOG_PATH          = os.path.join(TRAINING_DIR, "log.json")
YAML_PATH         = os.path.join(TRAINING_DIR, "data.yaml")
RETRAIN_THRESHOLD = 50   # Auto-retrain after this many new approved samples

# Map class names → YOLO class index
# Matches actual model outputs:
#   pole_model.pt    → indices 0–1  (main_pole, strut_pole)
#   insulator_new.pt → indices 2–5  (HT_disc, HT_pin, LT_pin, shackle_insulator)
#   best_component.pt → indices 6–16 (hardware components — 11 classes)
#   cable_best.pt    → index 17     (conductor / cable)
CLASS_MAP = {
    # ── Pole model (pole_finetune.pt) ──────────────────────────────
    "POLE":                0,
    "MAIN_POLE":           0,
    "STRUT_POLE":          1,

    # ── Insulator model (insulator_new.pt) ──────────────────────
    "HT_DISC":             2,
    "HT_PIN":              3,
    "LT_PIN":              4,
    "SHACKLE_INSULATOR":   5,
    "INSULATOR":           3,

    # ── Hardware component model (best_component.pt, 11 classes) ─
    "V_CROSS_ARM":         6,
    "V_CROSS":             6,
    "TAPPING_ARM":         7,
    "TAPPING_CHANNEL":     7,
    "SIDE_ARM":            8,
    "SIDE_ARM_CHANNEL":    8,
    "T_RISING":            9,
    "TOP_CLEAT":           10,
    "SPECIAL_CLAMP":       11,
    "STAY_SET":            12,
    "AB_SWITCH":           13,
    "STREET_LIGHT":        14,
    "BOX_ARM":             15,
    "DTR":                 16,
    "DTR_TANK":            16,

    # ── Cable model (cable_best.pt) ─────────────────────────────
    "CONDUCTOR":           17,

    # ── Catch-all fallback ───────────────────────────────────────
    "OBJECT":              18,
}


# ─── Setup ────────────────────────────────────────────────────────────────────
def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    _write_yaml()

def _write_yaml():
    """Write/update YOLO data.yaml with current class list.
    CLASS_MAP may have aliases (multiple keys → same index); use the
    canonical name (first key seen for each index, ignoring duplicates).
    """
    # Build index→name, keeping only the first key that maps to each index
    idx_to_name = {}
    for name, idx in CLASS_MAP.items():
        if idx not in idx_to_name:
            idx_to_name[idx] = name
    nc = max(idx_to_name.keys()) + 1
    lines = [
        f"path: {os.path.abspath(TRAINING_DIR)}",
        f"train: images",
        f"val: images",
        f"nc: {nc}",
        "names:",
    ] + [f"  {i}: {idx_to_name.get(i, 'unknown')}" for i in range(nc)]
    with open(YAML_PATH, "w") as f:
        f.write("\n".join(lines))


# ─── DB Helpers ───────────────────────────────────────────────────────────────
def _get_conn():
    return get_db_connection()


def _init_training_table():
    init_postgres_db(seed_defaults=False)

# ─── Core Export Function ─────────────────────────────────────────────────────
def export_asset_to_training(asset_id: str, approved_by: str) -> dict:
    """
    Called when admin approves an asset.
    Returns a summary dict with counts.
    """
    ensure_dirs()
    _init_training_table()

    conn = _get_conn()
    image_rows = conn.execute(
        "SELECT * FROM asset_images WHERE asset_id = ?", (asset_id,)
    ).fetchall()
    conn.close()

    if not image_rows:
        return {"exported": 0, "classes": {}}

    total_exported = 0
    total_class_counts = {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    from db import hydrate_asset_image_record
    for row in image_rows:
        try:
            hydrated = hydrate_asset_image_record(row)
            detections = hydrated.get("detections", [])
            image_b64  = hydrated.get("image_b64")

            if not detections or not image_b64:
                continue

            # Decode image to get real dimensions
            img_bytes = base64.b64decode(image_b64)
            import numpy as np, cv2
            nparr = np.frombuffer(img_bytes, np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]

            # 1. Identify all main poles in detections to use as anchor structures
            main_poles = [det for det in detections if "pole" in det.get("label", "").lower() and "strut" not in det.get("label", "").lower() and det.get("confirmed", True)]

            crops = []  # List of tuples: (cx1, cy1, cx2, cy2, crop_suffix)
            if main_poles:
                for idx, p_det in enumerate(main_poles):
                    px1, py1, px2, py2 = p_det["bbox"]
                    pw = px2 - px1
                    ph = py2 - py1
                    
                    assoc_dets = []
                    for det in detections:
                        if not det.get("confirmed", True) or det is p_det:
                            continue
                        bbox = det.get("bbox")
                        if not bbox or len(bbox) < 4:
                            continue
                        bx1, by1, bx2, by2 = bbox
                        
                        # Association checks:
                        # - Component is close horizontally (within 3.5x pole width from center)
                        # - Component lies vertically within the pole top region (from top - 150px to 75% height)
                        p_cx = (px1 + px2) / 2
                        b_cx = (bx1 + bx2) / 2
                        horiz_dist = abs(b_cx - p_cx)
                        is_close_x = horiz_dist < max(150, pw * 3.5)
                        is_in_y = (py1 - 150) <= by2 and by1 <= (py1 + ph * 0.75)
                        
                        if is_close_x and is_in_y:
                            assoc_dets.append(det)
                            
                    if assoc_dets:
                        # --- Stage 1: Structural Crop ---
                        # Filter out conductors/wires and poles from crop anchors to avoid over-expansion
                        hardware_dets = [det for det in assoc_dets if "conductor" not in det.get("label", "").lower() and "cable" not in det.get("label", "").lower() and "pole" not in det.get("label", "").lower()]
                        
                        # Pole top region box: Top 25% of the pole to serve as vertical anchor
                        pole_top_box = [px1, py1, px2, py1 + int(ph * 0.25)]
                        
                        # Merge hardware boxes with pole top anchor
                        anchor_boxes = [det["bbox"] for det in hardware_dets] + [pole_top_box]
                        
                        min_x = min(box[0] for box in anchor_boxes)
                        max_x = max(box[2] for box in anchor_boxes)
                        min_y = min(box[1] for box in anchor_boxes)
                        max_y = max(box[3] for box in anchor_boxes)
                        
                        # Horizontal Expansion using cross-arm spread (minimum of 150px padding)
                        pad_x = int(max(150, (max_x - min_x) * 0.15))
                        cx1 = max(0, min_x - pad_x)
                        cx2 = min(w, max_x + pad_x)
                        
                        # Vertical Expansion around hardware cluster
                        pad_y_top = 150
                        pad_y_bot = 100
                        cy1 = max(0, min_y - pad_y_top)
                        cy2 = min(h, max_y + pad_y_bot)
                        
                        # Aspect Ratio Clamping to prevent ultra-tall vertical crops
                        cw = cx2 - cx1
                        ch = cy2 - cy1
                        if cw > 10 and ch > 1.8 * cw:
                            # Centered clamping to keep hardware cluster in center
                            new_ch = int(1.8 * cw)
                            cy_center = (min_y + max_y) / 2
                            cy1 = max(0, int(cy_center - new_ch / 2))
                            cy2 = min(h, cy1 + new_ch)
                            # Re-align if hit bottom boundary
                            cy1 = max(0, cy2 - new_ch)
                            
                        crops.append((cx1, cy1, cx2, cy2, f"p{idx}"))

                        # --- Stage 2: Hardware-Head Crop (Tight component zoom) ---
                        head_classes = ["insulator", "insulators", "cleat", "clamp", "arm", "cross"]
                        head_dets = [det for det in assoc_dets if any(kw in det.get("label", "").lower() for kw in head_classes)]
                        
                        if head_dets:
                            hx1 = min(det["bbox"][0] for det in head_dets)
                            hy1 = min(det["bbox"][1] for det in head_dets)
                            hx2 = max(det["bbox"][2] for det in head_dets)
                            hy2 = max(det["bbox"][3] for det in head_dets)
                            
                            hw = hx2 - hx1
                            hh = hy2 - hy1
                            
                            # Expand by tight 30% margin
                            pad_hw_x = int(max(80, hw * 0.30))
                            pad_hw_y = int(max(60, hh * 0.30))
                            
                            hcx1 = max(0, hx1 - pad_hw_x)
                            hcx2 = min(w, hx2 + pad_hw_x)
                            hcy1 = max(0, hy1 - pad_hw_y)
                            hcy2 = min(h, hy2 + pad_hw_y)
                            
                            crops.append((hcx1, hcy1, hcx2, hcy2, f"hw{idx}"))
                    else:
                        cx1 = max(0, px1 - int(pw * 2.5))
                        cx2 = min(w, px2 + int(pw * 2.5))
                        cy1 = max(0, py1 - 150)
                        cy2 = min(h, py1 + int(ph * 0.25) + 100)
                        crops.append((cx1, cy1, cx2, cy2, f"p{idx}"))
            else:
                # Fallback to whole image if no main pole detected
                crops.append((0, 0, w, h, "full"))

            for cx1, cy1, cx2, cy2, suffix in crops:
                cw = cx2 - cx1
                ch = cy2 - cy1
                if cw <= 10 or ch <= 10:
                    continue

                crop_img = img[cy1:cy2, cx1:cx2]

                # Generate unique filename for this crop
                sample_id  = f"{str(uuid.uuid4())[:12]}_{suffix}"
                img_file   = f"{sample_id}.jpg"
                label_file = f"{sample_id}.txt"
                img_path   = os.path.join(IMAGES_DIR, img_file)
                lbl_path   = os.path.join(LABELS_DIR, label_file)

                # Save cropped image segment
                cv2.imwrite(img_path, crop_img)

                # Convert remapped detections inside this crop window to standard YOLO segmentation format
                yolo_lines  = []
                class_counts = {}

                for det in detections:
                    if not det.get("confirmed", True):
                        continue

                    label = det.get("label", "OBJECT").upper().replace(" ", "_")
                    bbox  = det.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                        
                    bx1, by1, bx2, by2 = bbox
                    
                    # Compute intersection with the crop window
                    ix1 = max(cx1, bx1)
                    iy1 = max(cy1, by1)
                    ix2 = min(cx2, bx2)
                    iy2 = min(cy2, by2)
                    
                    # If entirely outside the crop window, skip
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                        
                    # Proximity check: require at least 15% of detection overlap area or a valid centroid
                    det_area = (bx2 - bx1) * (by2 - by1)
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area < 0.15 * det_area:
                        # Exclude highly cut-off or mostly external objects
                        continue

                    class_id = CLASS_MAP.get(label, CLASS_MAP.get("OBJECT"))

                    # Retrieve polygon coordinate list if exists, otherwise construct 4 corners of intersection bbox
                    poly_pts = det.get("polygon")
                    mapped_poly = []
                    
                    if poly_pts and len(poly_pts) >= 3:
                        # Remap original polygon coordinates to cropped window and clip boundaries
                        for pt in poly_pts:
                            if len(pt) >= 2:
                                px, py = pt[0], pt[1]
                                clamped_x = max(cx1, min(cx2, px))
                                clamped_y = max(cy1, min(cy2, py))
                                mapped_poly.append((clamped_x - cx1, clamped_y - cy1))
                                
                        # Remove duplicates
                        cleaned = []
                        for p in mapped_poly:
                            if not cleaned or cleaned[-1] != p:
                                cleaned.append(p)
                        mapped_poly = cleaned
                        
                    # Fallback to bbox corners if polygon is invalid or has too few vertices
                    if len(mapped_poly) < 3:
                        mapped_poly = [
                            (ix1 - cx1, iy1 - cy1),
                            (ix2 - cx1, iy1 - cy1),
                            (ix2 - cx1, iy2 - cy1),
                            (ix1 - cx1, iy2 - cy1)
                        ]

                    # Normalize all mapped coordinates relative to the crop dimensions [0.0, 1.0]
                    norm_coords = []
                    for kx, ky in mapped_poly:
                        norm_coords.append(f"{kx / cw:.6f} {ky / ch:.6f}")
                        
                    # Standard YOLO segmentation format: class_id x1 y1 x2 y2 ...
                    yolo_lines.append(f"{class_id} " + " ".join(norm_coords))
                    class_counts[label] = class_counts.get(label, 0) + 1
                    total_class_counts[label] = total_class_counts.get(label, 0) + 1

                # Handle potential Negative Samples (blank label file for background training)
                if not yolo_lines:
                    import random
                    if random.random() > 0.10:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                        continue
                    else:
                        open(lbl_path, "w").close()
                        print(f"[TrainingPipeline] Exported background sample: {img_file}")
                else:
                    with open(lbl_path, "w") as f:
                        f.write("\n".join(yolo_lines))

                # Log to DB
                conn2 = _get_conn()
                conn2.execute("""
                    INSERT INTO training_samples
                        (id, asset_id, image_file, label_file, class_counts, approved_by, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (sample_id, asset_id, img_file, label_file,
                      json.dumps(class_counts), approved_by, timestamp))
                conn2.commit()
                conn2.close()

                total_exported += 1

        except Exception as e:
            print(f"[TrainingPipeline] Error exporting image: {e}")
            continue

    # Update log file
    _update_log(asset_id, total_exported, total_class_counts, approved_by)

    # Check if we should auto-trigger retrain
    total_new_samples = _count_pending_samples()
    retrain_queued = False
    if total_new_samples >= RETRAIN_THRESHOLD:
        retrain_queued = _queue_retrain(total_new_samples)

    return {
        "exported": total_exported,
        "classes": total_class_counts,
        "total_pool": total_new_samples,
        "retrain_queued": retrain_queued
    }

# ─── Training Stats ───────────────────────────────────────────────────────────
def get_training_stats() -> dict:
    """Returns stats for the Admin dashboard Training Pool panel."""
    _init_training_table()
    conn = _get_conn()

    samples = conn.execute(
        "SELECT class_counts, timestamp FROM training_samples ORDER BY timestamp DESC"
    ).fetchall()

    runs = conn.execute(
        "SELECT * FROM training_runs ORDER BY triggered_at DESC LIMIT 5"
    ).fetchall()

    conn.close()

    # Aggregate class counts
    total_by_class = {}
    for s in samples:
        counts = json.loads(s["class_counts"])
        for k, v in counts.items():
            total_by_class[k] = total_by_class.get(k, 0) + v

    # Check classes below threshold
    weak_classes = [c for c, cnt in total_by_class.items() if cnt < 300]

    return {
        "total_samples":   len(samples),
        "by_class":        total_by_class,
        "weak_classes":    weak_classes,
        "retrain_needed":  len(samples) >= RETRAIN_THRESHOLD,
        "threshold":       RETRAIN_THRESHOLD,
        "recent_runs":     [dict(r) for r in runs],
        "last_approved":   samples[0]["timestamp"] if samples else None,
    }

# ─── Retrain Trigger ──────────────────────────────────────────────────────────
def _count_pending_samples() -> int:
    conn = _get_conn()
    # Count samples added since last successful retrain
    last_run = conn.execute(
        "SELECT triggered_at FROM training_runs WHERE status = 'done' ORDER BY triggered_at DESC LIMIT 1"
    ).fetchone()
    conn.close()

    if last_run:
        conn2 = _get_conn()
        count = conn2.execute(
            "SELECT COUNT(*) as c FROM training_samples WHERE timestamp > ?",
            (last_run["triggered_at"],)
        ).fetchone()["c"]
        conn2.close()
        return count
    else:
        conn2 = _get_conn()
        count = conn2.execute("SELECT COUNT(*) as c FROM training_samples").fetchone()["c"]
        conn2.close()
        return count

def _queue_retrain(sample_count: int) -> bool:
    """Queue a retrain entry in the DB. Actual training started separately."""
    try:
        conn = _get_conn()
        # Avoid duplicate queue entries
        pending = conn.execute(
            "SELECT COUNT(*) as c FROM training_runs WHERE status = 'queued'"
        ).fetchone()["c"]

        if pending == 0:
            conn.execute("""
                INSERT INTO training_runs (triggered_at, sample_count, status)
                VALUES (?, ?, 'queued')
            """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sample_count))
            conn.commit()
            print(f"[TrainingPipeline] Retrain queued with {sample_count} samples.")
        conn.close()
        return True
    except Exception as e:
        print(f"[TrainingPipeline] Queue error: {e}")
        return False

def _update_log(asset_id, exported, class_counts, approved_by):
    log = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH) as f:
                log = json.load(f)
        except:
            log = []
    log.append({
        "asset_id":    asset_id,
        "exported":    exported,
        "classes":     class_counts,
        "approved_by": approved_by,
        "timestamp":   datetime.now().isoformat()
    })
    with open(LOG_PATH, "w") as f:
        json.dump(log[-500:], f, indent=2)  # Keep last 500 entries

# ─── Initialize on import ─────────────────────────────────────────────────────
ensure_dirs()
_init_training_table()





import ast
from collections import Counter, defaultdict
import time
import json
import os
import pathlib

TOTAL_ANNOTATED_DATASET_ROOT = "total_annotated_dataset"
if os.path.isdir(TOTAL_ANNOTATED_DATASET_ROOT):
    DATASET_STATS_ROOTS = [
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "components_dataset"),
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "insulator_dataset"),
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "insulator_self", "insulator_yolo"),
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "master_dataset_comp"),
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "Pole_dataset", "dataset_annotated", "dataset 1"),
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "Pole_dataset", "dataset_annotated", "dataset 2"),
    ]
    DATASET_INVENTORY_ROOTS = [
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "components_dataset"),
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "insulator_dataset"),
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "insulator_self"),
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "master_dataset_comp"),
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "Pole_dataset"),
    ]
    DATASET_IMAGE_STATS_ROOTS = [
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "components_dataset"),
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "insulator_self", "insulator_yolo"),
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "master_dataset_comp"),
        os.path.join(TOTAL_ANNOTATED_DATASET_ROOT, "Pole_dataset"),
    ]
else:
    DATASET_STATS_ROOTS = ["training_data"]
    DATASET_INVENTORY_ROOTS = DATASET_STATS_ROOTS
    DATASET_IMAGE_STATS_ROOTS = DATASET_STATS_ROOTS

DEFAULT_POLE_DATASET_CLASS_MAP = {0: "MAIN_POLE", 1: "STRUT_POLE"}
DEFAULT_INSULATOR_DATASET_CLASS_MAP = {0: "HT_DISC", 1: "HT_PIN", 2: "LT_PIN", 3: "SHACKLE_INSULATOR"}
CHANNEL_DATASET_CLASS_MAP = {
    0: "INSULATOR",
    1: "V_CROSS_ARM",
    2: "TAPPING_ARM",
    3: "TOP_CLEAT",
    4: "SIDE_ARM",
    5: "T_RISING",
    6: "SPECIAL_CLAMP",
    7: "STAY_SET",
    8: "STREET_LIGHT",
    9: "BOX_ARM",
    10: "DTR",
}
COMPONENT_DATASET_CLASS_MAP = CHANNEL_DATASET_CLASS_MAP
HARDWARE_DATASET_CLASS_MAP = CHANNEL_DATASET_CLASS_MAP
EXCLUDED_STATS_CLASSES = {"HORN_GAP_FUSE", "GO_SWITCH", "OTHERS"}

def _count_dataset_files(folder):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_count = 0
    annotation_count = 0
    if not os.path.isdir(folder):
        return image_count, annotation_count
    for root, _, files in os.walk(folder):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_exts:
                image_count += 1
            elif ext == ".txt":
                label_path = os.path.join(root, filename)
                try:
                    with open(label_path, "r", encoding="utf-8", errors="ignore") as handle:
                        for line in handle:
                            if line.strip():
                                annotation_count += 1
                except Exception:
                    pass
            elif ext == ".json":
                annotation_count += 1
    return image_count, annotation_count

def _normalise_stats_label(label):
    return str(label or "OBJECT").strip().upper().replace(" ", "_")

def _parse_dataset_names_yaml(yaml_path):
    if not yaml_path or not os.path.exists(yaml_path):
        return {}
    try:
        lines = pathlib.Path(yaml_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as exc:
        print(f"[TRAINING-STATS] Could not read class map {yaml_path}: {exc}", flush=True)
        return {}
    names = {}
    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped.startswith("names"):
            continue
        inline = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
        if inline:
            try:
                parsed = ast.literal_eval(inline)
                if isinstance(parsed, list):
                    return {i: _normalise_stats_label(name) for i, name in enumerate(parsed)}
                if isinstance(parsed, dict):
                    return {int(k): _normalise_stats_label(v) for k, v in parsed.items()}
            except Exception:
                pass
        for child in lines[idx + 1:]:
            if not child.startswith((" ", "	")):
                break
            child = child.strip()
            if ":" not in child:
                continue
            key, value = child.split(":", 1)
            try:
                names[int(key.strip())] = _normalise_stats_label(value.strip().strip("\'\""))
            except ValueError:
                continue
        break
    return names

def _is_placeholder_class_map(class_map):
    if not class_map:
        return False
    values = [str(value).strip().upper() for value in class_map.values()]
    return all(value.startswith("CLASS_") for value in values)


def _dataset_class_map_for(folder):
    normalized = folder.replace("\\", "/")

    if "Pole_dataset/dataset_annotated/dataset 1" in normalized:
        return {0: "MAIN_POLE", 1: "STRUT_POLE"}
    if "Pole_dataset/dataset_annotated/dataset 2" in normalized:
        return {0: "MAIN_POLE", 1: "STRUT_POLE"}
    if "components_dataset" in normalized:
        return CHANNEL_DATASET_CLASS_MAP
    if "insulator_dataset" in normalized or "insulator_self" in normalized:
        return DEFAULT_INSULATOR_DATASET_CLASS_MAP
    if "master_dataset_comp" in normalized:
        return CHANNEL_DATASET_CLASS_MAP

    folder_name = os.path.basename(os.path.normpath(folder)).lower()
    if folder_name == "training_data_component":
        return COMPONENT_DATASET_CLASS_MAP
    if folder_name in {"dataset 1 labels&images", "dataset 2 labels&images"}:
        return {0: "MAIN_POLE", 1: "STRUT_POLE"}
    if folder_name in {"raw_dataset", "training_data"}:
        return {0: "MAIN_POLE", 1: "MAIN_POLE"}
    if folder_name == "training_data_hardware":
        return HARDWARE_DATASET_CLASS_MAP
    if folder_name in {"dataset_channels", "dataset_combined"}:
        return CHANNEL_DATASET_CLASS_MAP

    yaml_map = _parse_dataset_names_yaml(os.path.join(folder, "data.yaml"))
    if yaml_map and not _is_placeholder_class_map(yaml_map):
        return yaml_map
    return {}

def _scan_dataset_label_stats(folder):
    class_map = _dataset_class_map_for(folder)
    normalized_folder = folder.replace("\\", "/")
    count_class_presence_per_file = "Pole_dataset/dataset_annotated" in normalized_folder
    by_class = Counter()
    images_per_class = Counter()
    unknown_ids = Counter()
    label_files = 0
    if not os.path.isdir(folder):
        return by_class, images_per_class, unknown_ids, label_files
    for root, _, files in os.walk(folder):
        for filename in files:
            if os.path.splitext(filename)[1].lower() != ".txt":
                continue
            label_files += 1
            labels_in_file = set()
            label_path = os.path.join(root, filename)
            try:
                with open(label_path, "r", encoding="utf-8", errors="ignore") as handle:
                    for raw_line in handle:
                        parts = raw_line.strip().split()
                        if not parts:
                            continue
                        try:
                            class_id = int(float(parts[0]))
                        except ValueError:
                            continue
                        label = class_map.get(class_id)
                        if not label:
                            unknown_ids[class_id] += 1
                            continue
                        if not count_class_presence_per_file:
                            by_class[label] += 1
                        labels_in_file.add(label)
            except Exception as exc:
                print(f"[TRAINING-STATS] Could not scan label file {label_path}: {exc}", flush=True)
                continue
            if count_class_presence_per_file:
                for label in labels_in_file:
                    by_class[label] += 1
            for label in labels_in_file:
                images_per_class[label] += 1
    return by_class, images_per_class, unknown_ids, label_files

_STATS_CACHE = None
_STATS_CACHE_TIME = 0.0
_STATS_CACHE_TTL = 600.0


def _calculate_actual_confidences():
    class_confs = defaultdict(list)
    try:
        db = get_db_connection()
        cur = db.execute(
            """
            SELECT i.detections
            FROM assets a
            LEFT JOIN asset_images i ON i.id = (
                SELECT id FROM asset_images WHERE asset_id = a.id LIMIT 1
            )
            WHERE a.status = 'approved'
            """
        )
        rows = cur.fetchall()
        db.close()
        for row in rows:
            dets_data = row.get('detections') if isinstance(row, dict) else row['detections']
            if not dets_data:
                continue
            dets = parse_db_json(dets_data)
            if not isinstance(dets, list):
                continue
            for det in dets:
                label = _normalise_stats_label(det.get('label', ''))
                if label == 'POLE':
                    label = 'MAIN_POLE'
                elif label == 'V_CROSS':
                    label = 'V_CROSS_ARM'
                elif label in {'INS_PIN', 'INSULATOR'}:
                    label = 'INSULATOR'
                elif label == 'TAPPING_ARM':
                    label = 'TAPPING_ARM'
                elif label == 'SIDE_ARM':
                    label = 'SIDE_ARM'
                elif label == 'T_RISING':
                    label = 'T_RISING'
                conf = det.get('confidence')
                if conf is not None:
                    try:
                        class_confs[label].append(float(conf))
                    except (TypeError, ValueError):
                        pass
    except Exception as exc:
        print(f"[TRAINING-STATS] Error calculating database confidences: {exc}", flush=True)

    def _weighted_conf(confs):
        if not confs:
            return None
        sum_c = sum(confs)
        if sum_c <= 0:
            return 0.0
        return sum(c * c for c in confs) / sum_c

    class_avg = {}
    all_confs = []
    for label, confs in class_confs.items():
        if confs:
            w_score = _weighted_conf(confs)
            if w_score is not None:
                class_avg[label] = w_score
            all_confs.extend(confs)

    overall_avg = _weighted_conf(all_confs)
    return overall_avg, class_avg


def _calculate_production_confidence_score(precision, recall, map50, map5095):
    metric_values = (precision, recall, map50, map5095)
    if any(value is None for value in metric_values):
        return None

    score = (
        0.45 * float(precision)
        + 0.45 * float(map50)
        + 0.05 * float(recall)
        + 0.05 * float(map5095)
    )
    return score


def build_training_dashboard_stats(stats=None):
    global _STATS_CACHE, _STATS_CACHE_TIME
    now = time.time()
    if _STATS_CACHE is not None and (now - _STATS_CACHE_TIME) < _STATS_CACHE_TTL:
        return _STATS_CACHE

    stats = dict(stats or {})
    dataset_by_class = Counter()
    dataset_images_per_class = Counter()
    unknown_by_dataset = {}
    scanned_label_files = 0
    stats_folder_annotations = {}

    for folder in DATASET_STATS_ROOTS:
        by_class, images_per_class, unknown_ids, label_files = _scan_dataset_label_stats(folder)
        for excluded_label in EXCLUDED_STATS_CLASSES:
            by_class.pop(excluded_label, None)
            images_per_class.pop(excluded_label, None)
        dataset_by_class.update(by_class)
        dataset_images_per_class.update(images_per_class)
        scanned_label_files += label_files
        stats_folder_annotations[folder] = sum(by_class.values())
        if unknown_ids:
            unknown_by_dataset[folder] = {str(k): v for k, v in sorted(unknown_ids.items())}
            print(f"[TRAINING-STATS] {folder} has label ids missing from its class map: {dict(sorted(unknown_ids.items()))}", flush=True)

    datasets = []
    for folder in DATASET_INVENTORY_ROOTS:
        if os.path.isdir(folder):
            images, annotations = _count_dataset_files(folder)
            included_stats_roots = [
                stats_root for stats_root in DATASET_STATS_ROOTS
                if stats_root == folder or stats_root.startswith(folder + os.sep)
            ]
            if included_stats_roots:
                annotations = sum(stats_folder_annotations.get(stats_root, 0) for stats_root in included_stats_roots)
            datasets.append({
                'name': folder,
                'path': os.path.abspath(folder),
                'count': images,
                'images': images,
                'annotations': annotations,
                'included_in_stats': bool(included_stats_roots),
            })

    by_class = dict(sorted(dataset_by_class.items(), key=lambda item: item[1], reverse=True))
    stats['total_samples'] = sum(_count_dataset_files(folder)[0] for folder in DATASET_IMAGE_STATS_ROOTS)
    stats['by_class'] = by_class
    stats['total_classes'] = len(by_class)
    stats['total_annotations'] = int(sum(by_class.values()))

    overall_avg, class_avg = _calculate_actual_confidences()
    stats['overall_avg_confidence'] = float(overall_avg) if overall_avg is not None else None
    stats['per_class_confidence'] = {label: float(conf) for label, conf in class_avg.items()}
    stats['avg_confidence'] = stats['overall_avg_confidence']
    stats['class_confidence'] = stats['per_class_confidence']
    stats["production_confidence_score"] = None

    stats['images_per_class'] = dict(dataset_images_per_class)
    stats['datasets'] = datasets
    stats['stats_source'] = 'dataset_folders'
    stats['scanned_label_files'] = scanned_label_files
    stats['unknown_label_ids'] = unknown_by_dataset

    metrics_path = os.path.join('static', 'model_metrics.json')
    validation_metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as handle:
                validation_metrics = json.load(handle)
        except Exception as exc:
            print(f"[app] Error reading model_metrics.json: {exc}", flush=True)
    stats['validation_metrics'] = validation_metrics

    if validation_metrics:
        p_vals = []
        r_vals = []
        m50_vals = []
        m5095_vals = []
        class_val_conf = {}

        for _model_key, model_stats in validation_metrics.items():
            if isinstance(model_stats, dict):
                if 'precision' in model_stats:
                    p_vals.append(model_stats['precision'])
                if 'recall' in model_stats:
                    r_vals.append(model_stats['recall'])
                if 'map50' in model_stats:
                    m50_vals.append(model_stats['map50'])
                if 'map5095' in model_stats:
                    m5095_vals.append(model_stats['map5095'])

                if 'class_metrics' in model_stats and isinstance(model_stats['class_metrics'], dict):
                    for c_name, c_metrics in model_stats['class_metrics'].items():
                        if isinstance(c_metrics, dict):
                            c_score = _calculate_production_confidence_score(
                                c_metrics.get('precision'),
                                c_metrics.get('recall'),
                                c_metrics.get('map50'),
                                c_metrics.get('map5095'),
                            )
                            if c_score is not None:
                                norm_name = c_name.upper().replace(' ', '_')
                                class_val_conf[norm_name] = c_score
                                if norm_name in {'INSULATORS', 'INSULATOR'}:
                                    for ins_key in ('INSULATOR', 'INSULATORS', 'HT_PIN', 'LT_PIN', 'PIN', 'HT_DISC', 'LT_DISC', 'DISC', 'SHACKLE', 'SHACKLE_INSULATOR'):
                                        class_val_conf[ins_key] = c_score
                                elif norm_name == 'TAPPING_ARM':
                                    class_val_conf['TAPPING_CHANNEL'] = c_score

        if p_vals:
            stats['overall_precision'] = sum(p_vals) / len(p_vals)
            stats['overall_recall'] = sum(r_vals) / len(r_vals)
            stats['overall_map50'] = sum(m50_vals) / len(m50_vals)
            stats['overall_map5095'] = sum(m5095_vals) / len(m5095_vals)
            
            prod_score = _calculate_production_confidence_score(
                stats["overall_precision"],
                stats["overall_recall"],
                stats["overall_map50"],
                stats["overall_map5095"],
            )
            stats["production_confidence_score"] = prod_score
            if prod_score is not None:
                stats['overall_avg_confidence'] = float(prod_score)
                stats['avg_confidence'] = stats['overall_avg_confidence']

        if class_val_conf:
            for k, v in class_val_conf.items():
                stats['per_class_confidence'][k] = float(v)
            stats['class_confidence'] = stats['per_class_confidence']

    model_descriptions = {
        'pole': 'Pole and strut pole detector',
        'components': 'Hardware and component detector',
        'insulator': 'Insulator detector',
        'shed': 'Shed counter',
        'conductor_unet': 'Conductor segmentation model',
        'video_component': 'Video pole detector',
    }
    model_paths = {
        'pole': 'models/pole_best6.pt',
        'components': 'models/channel_12class_v2.pt',
        'insulator': 'models/insulator_new.pt',
        'shed': 'models/shed_model.pt',
        'conductor_unet': 'models/cable_best.pt',
        'video_component': 'models/pole_best6.pt',
    }
    models = []
    for key, model_path in model_paths.items():
        exists = os.path.exists(model_path)
        models.append({
            'key': key,
            'label': key.replace('_', ' ').title(),
            'description': model_descriptions.get(key, ''),
            'exists': exists,
            'size_mb': round(os.path.getsize(model_path) / (1024 * 1024), 1) if exists else None,
        })
    stats['models'] = models

    _STATS_CACHE = stats
    _STATS_CACHE_TIME = now
    return stats
