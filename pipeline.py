"""
pipeline.py
────────────
Runs the model stack and returns normalized raw detections.

The current mode intentionally avoids business-rule classification and
prefers direct model labels plus light response shaping for the UI.
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path

# Fix relative imports when running from root
curr_dir = Path(__file__).parent.absolute()
if str(curr_dir) not in sys.path:
    sys.path.append(str(curr_dir))

from typing import Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter

from ultralytics import YOLO

from config import (
    DETECTION_CONF, DETECTION_CONF_POLE, DETECTION_IOU,
    OBB_CLASS_KEYWORDS,
    THRESHOLD_INSULATOR, THRESHOLD_CROSSARM, THRESHOLD_POLE, THRESHOLD_CONDUCTOR,
    POLE_IDEAL_ANGLE_DEG,
    POLE_STAGE_IMGSZ, COMPONENT_STAGE_IMGSZ, INSULATOR_STAGE_IMGSZ, SHED_STAGE_IMGSZ, CONDUCTOR_STAGE_IMGSZ,
    meets_pole_confidence,
)

@dataclass
class InsulatorResult:
    box:              tuple         # (x1, y1, x2, y2)
    aspect_ratio:     float         # height / width
    type_heuristic:   str           # "pin" | "disc" | "uncertain"
    type_final:       str           # "pin" | "disc" | "shackle"
    type_confidence:  str           # "high" | "medium" | "low"
    shed_count:       int   = 0
    voltage:          str   = "unknown"
    detection_conf:   float = 0.0

    # Adjustment fault fields
    raw_angle_deg:    Optional[float] = None
    polygon:          Optional[list]  = None
    tilt_deg:         float = 0.0              # deviation from ideal vertical
    adjustment_fault: bool  = False            # True if misaligned
    fault_severity:   str   = "none"           # "none" | "warning" | "fault"
    fault_note:       str   = ""               # human-readable reason
    note:             str   = ""
    is_silhouette:    bool  = False            # True if ROI is too dark

@dataclass
class CrossarmResult:
    box:              tuple
    shape:            str    # "straight" | "v_arm" | "t_raising"
    confidence:       str    # "high" | "medium" | "low"
    aspect_ratio:     float
    raw_angle_deg:    Optional[float] = None
    polygon:          Optional[list]  = None
    detection_conf:   float = 0.0
    # Adjustment fault
    adjustment_fault: bool  = False
    fault_severity:   str   = "none"   # "none" | "warning" | "fault"
    tilt_deg:         float = 0.0
    fault_note:       str   = ""
    note:             str   = ""

@dataclass
class PoleOrientationResult:
    box:              tuple
    pole_type:        str    # "vertical_pole" | "strut_pole"
    lean_angle_deg:   float  # degrees from vertical
    confidence:       str
    polygon:          Optional[list]  = None
    detection_conf:   float = 0.0
    # Adjustment fault
    adjustment_fault: bool  = False
    fault_severity:   str   = "none"
    fault_note:       str   = ""
    note:             str   = ""


# ── Pipeline output ───────────────────────────────────────────

@dataclass
class PipelineResult:
    """Complete output of one inference run."""
    final_class:      str
    class_id:         int
    reason:           str
    voltage:          str
    confidence:       str
    signals_used:     list

    insulators:       list  = field(default_factory=list)
    all_poles:        list  = field(default_factory=list) # List of PoleOrientationResult
    pole_orientation: Optional[PoleOrientationResult] = None # Primary pole for rule engine
    crossarms:        list  = field(default_factory=list)
    conductors:       list  = field(default_factory=list) # Actual boxes
    street_lights:    list  = field(default_factory=list) # (box, conf, poly)
    others:           list  = field(default_factory=list) # (label, box, conf, poly)
    crossarm_shape:   str   = "none"
    crossarm_count:   int   = 0
    conductor_count:  int   = 0
    pole_id:          str   = "Not Found"
    flags:            dict  = field(default_factory=dict)

    # Adjustment fault summary
    adjustment_faults: list = field(default_factory=list)
    # Each entry: {"component": str, "severity": str, "note": str}




@dataclass
class LegacyComponentSignals:
    """Legacy compatibility stub formerly housed in rule_engine.py."""

    insulator_type: str = "unknown"
    insulator_voltage: str = "unknown"
    shed_count: int = 0
    insulator_conf: str = "low"
    has_dtr: bool = False
    has_ab_cable: bool = False
    has_lattice: bool = False
    has_jumper: bool = False
    has_ht_and_lt: bool = False
    has_broken_wire: bool = False
    has_vegetation: bool = False
    pole_type: str = "unknown"
    lean_angle_deg: float = 0.0
    crossarm_count: int = 0
    crossarm_shape: str = "unknown"
    conductor_count: int = 0


@dataclass
class LegacyClassificationResult:
    final_class: str
    class_id: int
    reason: str
    voltage: str
    confidence: str
    signals_used: list = field(default_factory=list)


def classify_pole(signals: LegacyComponentSignals) -> LegacyClassificationResult:
    return LegacyClassificationResult(
        final_class=signals.pole_type or "raw_mode",
        class_id=-1,
        reason="Legacy rule engine disabled; raw-prediction mode is active.",
        voltage="raw_mode",
        confidence=signals.insulator_conf or "low",
        signals_used=[],
    )

# ── Helper: keyword matcher ───────────────────────────────────

def _normalize_label(label: str, fallback: str = "object") -> str:
    cleaned = str(label or fallback).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or fallback


def _match_keyword(cls_name: str, category: str) -> bool:
    """
    Checks if a detected class name belongs to a category.
    Uses OBB_CLASS_KEYWORDS from config.
    """
    name_lower = cls_name.lower()
    
    # Safety guard: 'clamp' contains the substring 'lamp', but a clamp is never a lamp/street light
    if category == "lamp_head" and "clamp" in name_lower:
        return False
        
    return any(kw.lower() in name_lower for kw in OBB_CLASS_KEYWORDS.get(category, []))


# ── Main pipeline ─────────────────────────────────────────────

class InfrastructurePipeline:
    """
    Raw-prediction infrastructure inspection pipeline.
    Combines model detections and light normalization for the app.
    """

    def __init__(
        self,
        component_model_path: str,
        insulator_model_path: str,
        shed_model_path: Optional[str] = None,
        crossarm_model_path: Optional[str] = None,
        crop_classifier_path: Optional[str] = None,
        conf: float = DETECTION_CONF,
        iou:  float = DETECTION_IOU,
    ):
        """
        Args:
            component_model_path : path to your trained component YOLO (.pt)
            insulator_model_path : path to specialized insulator detection model (.pt)
            shed_model_path      : (Obsolete/Unused) path to your shed-count model (.pt)
            crossarm_model_path  : path to model for crossarms (e.g., best_whole.pt)
            crop_classifier_path : (Obsolete/Unused) path to optional insulator crop classifier (.pt)
            conf                 : detection confidence threshold
            iou                  : NMS IoU threshold
        """
        print("Loading component model...")
        self.component_model = YOLO(component_model_path)
        print("Loading dedicated insulator detector...")
        self.insulator_detector = YOLO(insulator_model_path)

        if crossarm_model_path and Path(crossarm_model_path).exists():
            print("Loading dedicated crossarm detector...")
            self.crossarm_detector = YOLO(crossarm_model_path)
        else:
            self.crossarm_detector = None

        if shed_model_path and Path(shed_model_path).exists():
            print("Loading dedicated shed detector...")
            self.shed_detector = YOLO(shed_model_path)
        else:
            self.shed_detector = None

        print("Running with raw predictions (bypassing secondary classifiers)...")

        self.conf = conf
        self.iou  = iou
        
        print("Pipeline ready.\n")

    def predict(
        self,
        image_path: str,
    ) -> PipelineResult:
        """
        Runs full pipeline on one image.

        Args:
            image_path : path to input image
        Returns:
            PipelineResult containing normalized raw model outputs
        """
        img_original = cv2.imread(image_path)
        if img_original is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img_h, img_w = img_original.shape[:2]

        import gc
        import torch

        stage_times = {}

        # ── Stage 1: Pole detector on full image ────────────────────────
        stage_start = time.perf_counter()
        raw_pole = self._run_pole_detector(image_path)
        stage_times["pole_detect"] = time.perf_counter() - stage_start

        insulator_boxes  = []
        pole_boxes_raw   = self._collect_pole_stage_detections(raw_pole)
        crossarm_boxes   = []
        conductor_boxes  = []
        street_light_boxes = []
        other_boxes        = []
        flags = defaultdict(bool)

        if not pole_boxes_raw:
            print(f"DEBUG: No pole detected above threshold {THRESHOLD_POLE:.2f}. Redirecting to no_pole_detected.")
            return PipelineResult(
                final_class      = "no_pole_detected",
                class_id         = -1,
                reason           = f"No pole detected above threshold {THRESHOLD_POLE:.2f}.",
                voltage          = "raw_mode",
                confidence       = "0.00",
                signals_used     = ["pole:0", "early_exit:no_pole_detected"],
                insulators       = [],
                all_poles        = [],
                pole_orientation = None,
                crossarms        = [],
                conductors       = [],
                street_lights    = [],
                others           = [],
                crossarm_shape   = "none",
                crossarm_count   = 0,
                conductor_count  = 0,
                flags            = dict(flags),
                adjustment_faults= [],
                pole_id          = "Not Found",
            )

        primary_pole_box = self._select_primary_pole_box(pole_boxes_raw)
        pole_crop_x1, pole_crop_y1, pole_crop_x2, pole_crop_y2 = self._generate_primary_pole_crop(img_original, primary_pole_box)
        pole_crop_img = img_original[pole_crop_y1:pole_crop_y2, pole_crop_x1:pole_crop_x2]

        # ── Stage 2: Components detector on pole crop ───────────────────
        stage_start = time.perf_counter()
        raw_components = self._run_component_detector(pole_crop_img)
        self._collect_component_stage_detections(
            raw_components,
            pole_crop_x1,
            pole_crop_y1,
            crossarm_boxes,
            street_light_boxes,
            other_boxes,
            flags,
        )
        stage_times["component_detect"] = time.perf_counter() - stage_start

        # ── Refine crop using component lower bound after pole detect ───
        stage_start = time.perf_counter()
        crop_x1, crop_y1, crop_x2, crop_y2 = self._generate_component_guided_crop(
            img_original,
            pole_boxes_raw,
            crossarm_boxes,
            insulator_boxes,
            other_boxes,
        )
        stage_times["component_crop"] = time.perf_counter() - stage_start
        crop_img = img_original[crop_y1:crop_y2, crop_x1:crop_x2]

        # ── Stage 3: Insulator detector on components crop ──────────────
        stage_start = time.perf_counter()
        raw_insulator = self._run_insulator_detector(crop_img)
        self._collect_insulator_stage_detections(raw_insulator, crop_x1, crop_y1, insulator_boxes)
        stage_times["insulator_detect"] = time.perf_counter() - stage_start

        # ── Stage 4: Shed detector on components crop ───────────────────
        stage_start = time.perf_counter()
        raw_shed = self._run_shed_detector(crop_img)
        self._collect_shed_stage_detections(raw_shed, crop_x1, crop_y1, other_boxes)
        stage_times["shed_detect"] = time.perf_counter() - stage_start
        print("[Stage Timing] " + ", ".join(f"{name}={duration:.3f}s" for name, duration in stage_times.items()))

        # ── Stage 5: Conductor stage remains separate in image pipeline ─

        # ── Deduplicate multi-scale detections (NMS) ─────────
        insulator_boxes = self._nms(insulator_boxes, iou_threshold=0.75) 
        pole_boxes_raw  = self._nms(pole_boxes_raw,  iou_threshold=0.45)
        crossarm_boxes  = self._nms(crossarm_boxes,  iou_threshold=0.45)
        conductor_boxes = self._nms(conductor_boxes, iou_threshold=0.65)
        street_light_boxes = self._nms(street_light_boxes, iou_threshold=0.45)
        other_boxes        = self._nms(other_boxes,        iou_threshold=0.45)

        # ── Infer missing pole logic ──────────────
        # DISABLED for strict 80% confidence requirement
        # if not pole_boxes_raw:
        #     inferred = self._infer_pole_if_missing(insulator_boxes, crossarm_boxes, img_h, img_w)
        #     ...

        global_tilt = 0.0

        # ── Normalize raw insulator detections ───────────────────────────
        insulator_results = []
        for box, conf_val, angle_deg, polygon, detected_cls in insulator_boxes:
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            ar = round(bh / bw if bw > 0 else 0, 2)
            raw_label = _normalize_label(detected_cls, "insulator")

            ins_result = InsulatorResult(
                box             = box,
                aspect_ratio    = ar,
                type_heuristic  = raw_label,
                type_final      = raw_label,
                type_confidence = "high" if conf_val > 0.5 else "medium",
                shed_count      = 0,
                voltage         = "unknown",
                detection_conf  = round(conf_val, 2),
                raw_angle_deg   = angle_deg,
                polygon         = polygon,
                tilt_deg        = 0.0,
                adjustment_fault= False,
                fault_severity  = "none",
                fault_note      = "",
                note            = f"Raw insulator class: {raw_label}",
                is_silhouette   = False,
            )
            insulator_results.append(ins_result)

        all_poles = []
        pole_result = None
        if pole_boxes_raw:
            pole_boxes_raw.sort(key=lambda x: (1 if x[4] else 0, -x[1]))

            print(f"DEBUG: Pole sort order: {[(('STRUT' if x[4] else 'MAIN'), f'conf={x[1]:.2f}', f'angle={x[2]}') for x in pole_boxes_raw]}")

            for i, (p_box, p_conf, p_angle, p_poly, p_is_strut) in enumerate(pole_boxes_raw):
                pole_type = "strut_pole" if p_is_strut else "pole"
                lean = round(abs(POLE_IDEAL_ANGLE_DEG - abs(p_angle)), 1) if p_angle is not None else 0.0

                pr = PoleOrientationResult(
                    box              = p_box,
                    pole_type        = pole_type,
                    lean_angle_deg   = lean,
                    confidence       = "high" if p_conf > 0.5 else "medium",
                    polygon         = p_poly,
                    detection_conf   = p_conf,
                    adjustment_fault = False,
                    fault_severity   = "none",
                    fault_note       = "",
                    note             = f"Raw pole detection: {pole_type}",
                )

                all_poles.append(pr)

                if i == 0:
                    pole_result = pr
                    print(f"DEBUG: Primary pole selected: type={pr.pole_type}, lean={pr.lean_angle_deg:.1f}°, conf={p_conf:.2f}")
        pole_id = "Not Found"

        crossarm_results = []
        for box, conf, ang, poly, native_cls in crossarm_boxes:
            cx1, cy1, cx2, cy2 = box
            c_w = cx2 - cx1
            c_h = cy2 - cy1
            c_ar = c_w / c_h if c_h > 0 else 0
            raw_shape = _normalize_label(native_cls, "crossarm")

            cr = CrossarmResult(
                box              = box,
                shape            = raw_shape,
                confidence       = "high" if conf > 0.5 else "medium",
                aspect_ratio     = round(c_ar, 2),
                raw_angle_deg    = ang,
                polygon          = poly,
                detection_conf   = conf,
                adjustment_fault = False,
                fault_severity   = "none",
                tilt_deg         = 0.0,
                fault_note       = "",
                note             = f"Raw crossarm class: {raw_shape}",
            )
            crossarm_results.append(cr)

        # ── Aggregate raw label summary ────────────────────────
        dominant_shape = crossarm_results[0].shape if crossarm_results else "none"
        n_crossarms = len(crossarm_results)
        adjustment_faults = []

        raw_counts = Counter()
        raw_confidences = defaultdict(list)

        def track(label, confidence):
            normalized = _normalize_label(label, "object")
            raw_counts[normalized] += 1
            raw_confidences[normalized].append(float(confidence))

        for ins in insulator_results:
            track(ins.type_heuristic, ins.detection_conf)

        for pole in all_poles:
            track(pole.pole_type, pole.detection_conf)

        for arm in crossarm_results:
            track(arm.shape, arm.detection_conf)

        for _, conf, _ in conductor_boxes:
            track("conductor", conf)

        for _, conf, _ in street_light_boxes:
            track("street_light", conf)

        for label, _, conf, _ in other_boxes:
            track(label, conf)

        dominant_label = "no_detection"
        dominant_conf = 0.0
        if raw_counts:
            ranked_labels = sorted(
                raw_counts.items(),
                key=lambda item: (
                    item[1],
                    sum(raw_confidences[item[0]]) / len(raw_confidences[item[0]]),
                    item[0] != "pole",
                ),
                reverse=True,
            )
            dominant_label = ranked_labels[0][0]
            dominant_conf_values = raw_confidences.get(dominant_label, [])
            dominant_conf = sum(dominant_conf_values) / len(dominant_conf_values) if dominant_conf_values else 0.0

        pipeline_result = PipelineResult(
            final_class      = dominant_label,
            class_id         = -1,
            reason           = "Raw detector outputs only.",
            voltage          = "raw_mode",
            confidence       = f"{dominant_conf:.2f}",
            signals_used     = [f"{label}:{count}" for label, count in raw_counts.most_common()],
            insulators       = insulator_results,
            all_poles        = all_poles,
            pole_orientation = pole_result,
            crossarms        = crossarm_results,
            conductors       = conductor_boxes,
            street_lights    = street_light_boxes,
            others           = other_boxes,
            crossarm_shape   = dominant_shape,
            crossarm_count   = n_crossarms,
            conductor_count  = len(conductor_boxes),
            flags            = dict(flags),
            adjustment_faults= adjustment_faults,
            pole_id          = pole_id,
        )

        return pipeline_result

    def _extract_stage_detections(self, model, raw_results, offset_x: int = 0, offset_y: int = 0):
        detections = []
        if raw_results is None:
            return detections

        for result in list(raw_results):
            boxes = result.boxes if hasattr(result, "boxes") and result.boxes else None
            masks = result.masks if hasattr(result, "masks") and result.masks else None
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                box_obj = boxes[i]
                cls_name = model.names[int(box_obj.cls)]
                conf_val = float(box_obj.conf)
                b = box_obj.xyxy[0].cpu().numpy()
                x1 = int(b[0]) + offset_x
                y1 = int(b[1]) + offset_y
                x2 = int(b[2]) + offset_x
                y2 = int(b[3]) + offset_y
                polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                mask_points = None

                if masks is not None and getattr(masks, "xy", None) is not None and i < len(masks.xy):
                    poly_pts = masks.xy[i]
                    if len(poly_pts) > 0:
                        polygon = [[int(pt[0] + offset_x), int(pt[1] + offset_y)] for pt in poly_pts]
                        mask_points = poly_pts

                detections.append({
                    "label": cls_name,
                    "conf": conf_val,
                    "box": (x1, y1, x2, y2),
                    "polygon": polygon,
                    "mask_points": mask_points,
                })

        return detections

    def _estimate_pole_angle(self, box, mask_points):
        if mask_points is None or len(mask_points) == 0:
            return None
        try:
            from skimage.morphology import skeletonize

            x1, y1, x2, y2 = box
            w, h = int(x2 - x1), int(y2 - y1)
            if w <= 0 or h <= 0:
                return None

            pole_mask = np.zeros((h, w), dtype=np.uint8)
            poly_pts = np.array(mask_points, dtype=np.int32).copy()
            poly_pts[:, 0] -= int(x1)
            poly_pts[:, 1] -= int(y1)
            cv2.fillPoly(pole_mask, [poly_pts], 1)
            skeleton = skeletonize(pole_mask > 0)
            y_skel, x_skel = np.nonzero(skeleton)
            if len(x_skel) <= 10:
                return None

            m, _ = np.polyfit(y_skel, x_skel, 1)
            return 90.0 - np.degrees(np.arctan(m))
        except Exception as exc:
            print(f"DEBUG: Pole angle estimation failed: {exc}")
            return None

    def _run_pole_detector(self, image_path: str):
        print("DEBUG: Running pole detector (stage 1)...")
        return self.component_model(image_path, conf=DETECTION_CONF_POLE, iou=self.iou, verbose=False, imgsz=POLE_STAGE_IMGSZ)

    def _collect_pole_stage_detections(self, raw_results):
        pole_boxes_raw = []
        for det in self._extract_stage_detections(self.component_model, raw_results):
            cls_name = det["label"]
            conf_val = det["conf"]
            if not _match_keyword(cls_name, "pole"):
                continue
            if not meets_pole_confidence(conf_val):
                print(f"DEBUG: Pole candidate rejected below threshold: label={cls_name}, conf={conf_val:.2f}, threshold={THRESHOLD_POLE:.2f}, box={det['box']}")
                continue

            print(f"DEBUG: Raw pole bbox accepted before crop padding: label={cls_name}, conf={conf_val:.2f}, box={det['box']}")
            angle_deg = self._estimate_pole_angle(det["box"], det.get("mask_points"))
            is_strut = ("strut" in cls_name.lower())
            if angle_deg is not None:
                skel_deviation = abs(90.0 - abs(angle_deg))
                if not is_strut and skel_deviation > 35.0:
                    print(f"🚫 [Skeleton Filter] Rejected {cls_name} (conf={conf_val:.2f}): skeleton {angle_deg:.1f}° deviates {skel_deviation:.1f}° from vertical")
                    angle_deg = None
                elif not is_strut and skel_deviation > 15.0:
                    is_strut = True

            self._categorise(
                cls_name, det["box"], conf_val, angle_deg,
                [], pole_boxes_raw, [], [], [], [], defaultdict(bool),
                polygon=det["polygon"], is_strut=is_strut
            )

        return pole_boxes_raw

    def _select_primary_pole_box(self, pole_boxes_raw):
        if not pole_boxes_raw:
            return None
        sorted_poles = sorted(pole_boxes_raw, key=lambda x: (1 if x[4] else 0, -x[1]))
        for pole_box, _, _, _, is_strut in sorted_poles:
            if not is_strut:
                return pole_box
        return sorted_poles[0][0]

    def _generate_primary_pole_crop(self, img: np.ndarray, primary_box: tuple) -> Tuple[int, int, int, int]:
        img_h, img_w = img.shape[:2]
        px1, py1, px2, py2 = primary_box
        pole_width = max(1, px2 - px1)
        pole_height = max(1, py2 - py1)
        top_y2 = py1 + int(pole_height * 0.58)
        max_pad_x = max(140, int(img_w * 0.12))
        pad_x = min(max_pad_x, max(80, int(pole_width * 1.2)))
        pad_top = min(60, max(20, int(pole_height * 0.03)))
        pad_bottom = min(140, max(60, int(pole_height * 0.10)))
        crop_x1 = max(0, px1 - pad_x)
        crop_y1 = max(0, py1 - pad_top)
        crop_x2 = min(img_w, px2 + pad_x)
        crop_y2 = min(img_h, top_y2 + pad_bottom)
        print(f"DEBUG: Primary pole crop generated: raw_box={primary_box}, pad_x={pad_x}, pad_top={pad_top}, pad_bottom={pad_bottom}, x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}")
        return crop_x1, crop_y1, crop_x2, crop_y2

    def _run_component_detector(self, crop_img: np.ndarray):
        if self.crossarm_detector is None or crop_img.size == 0:
            return []
        print("DEBUG: Running components detector (stage 2) on pole crop...")
        return self.crossarm_detector(crop_img, conf=THRESHOLD_CROSSARM, imgsz=COMPONENT_STAGE_IMGSZ, verbose=False)

    def _collect_component_stage_detections(self, raw_results, offset_x, offset_y, crossarm_boxes, street_light_boxes, other_boxes, flags):
        if self.crossarm_detector is None:
            return
        for det in self._extract_stage_detections(self.crossarm_detector, raw_results, offset_x=offset_x, offset_y=offset_y):
            cls_name = det["label"]
            conf_val = det["conf"]
            if _match_keyword(cls_name, "pole") or _match_keyword(cls_name, "insulator") or _match_keyword(cls_name, "conductor"):
                continue
            self._categorise(
                cls_name, det["box"], conf_val, None,
                [], [], crossarm_boxes, [], street_light_boxes, other_boxes, flags,
                polygon=det["polygon"], is_strut=False
            )

    def _run_insulator_detector(self, crop_img: np.ndarray):
        if crop_img.size == 0:
            return []
        print("DEBUG: Running insulator detector (stage 3) on components crop...")
        return self.insulator_detector(crop_img, conf=THRESHOLD_INSULATOR, imgsz=INSULATOR_STAGE_IMGSZ, verbose=False)

    def _collect_insulator_stage_detections(self, raw_results, offset_x, offset_y, insulator_boxes):
        for det in self._extract_stage_detections(self.insulator_detector, raw_results, offset_x=offset_x, offset_y=offset_y):
            cls_name = det["label"]
            conf_val = det["conf"]
            if not _match_keyword(cls_name, "insulator"):
                continue
            if cls_name == "LT_pin" and conf_val < 0.45:
                print(f"DEBUG: LT_pin insulator rejected (conf {conf_val:.2f} < 0.45)")
                continue
            insulator_boxes.append((det["box"], conf_val, None, det["polygon"], cls_name))

    def _run_shed_detector(self, crop_img: np.ndarray):
        if self.shed_detector is None or crop_img.size == 0:
            return []
        print("DEBUG: Running shed detector (stage 4) on components crop...")
        return self.shed_detector(crop_img, conf=THRESHOLD_INSULATOR, imgsz=SHED_STAGE_IMGSZ, verbose=False)

    def _collect_shed_stage_detections(self, raw_results, offset_x, offset_y, other_boxes):
        if self.shed_detector is None:
            return
        for det in self._extract_stage_detections(self.shed_detector, raw_results, offset_x=offset_x, offset_y=offset_y):
            cls_name = det["label"].upper()
            other_boxes.append((cls_name, det["box"], det["conf"], det["polygon"]))

    def _generate_component_guided_crop(
        self,
        img: np.ndarray,
        pole_boxes: list,
        crossarm_boxes: list,
        insulator_boxes: list,
        other_boxes: list,
    ) -> Tuple[int, int, int, int]:
        """
        Build a generalized crop for the insulator detector.

        Flow:
        1. pick the primary detected pole
        2. collect nearby structural components around that pole
        3. crop from the pole top down to the lowest component lower bound

        This keeps the insulator model focused on the useful pole-top region
        instead of the full frame.
        """
        img_h, img_w = img.shape[:2]

        if pole_boxes:
            sorted_poles = sorted(pole_boxes, key=lambda item: (1 if item[4] else 0, -item[1]))
            primary_box = None
            for pole_box, _, _, _, is_strut in sorted_poles:
                if not is_strut:
                    primary_box = pole_box
                    break
            if primary_box is None:
                primary_box = sorted_poles[0][0]

            px1, py1, px2, py2 = primary_box
            pole_width = max(1, px2 - px1)
            pole_height = max(1, py2 - py1)
            center_x = (px1 + px2) / 2
            allowed_margin_x = min(max(140, int(img_w * 0.12)), max(180, int(pole_width * 1.8)))

            component_boxes = []

            for box, *_ in crossarm_boxes:
                bx1, by1, bx2, by2 = box
                bx_center = (bx1 + bx2) / 2
                if abs(bx_center - center_x) <= allowed_margin_x:
                    component_boxes.append(box)

            for box, *_ in insulator_boxes:
                bx1, by1, bx2, by2 = box
                bx_center = (bx1 + bx2) / 2
                if abs(bx_center - center_x) <= allowed_margin_x:
                    component_boxes.append(box)

            for _, box, _, _ in other_boxes:
                bx1, by1, bx2, by2 = box
                bx_center = (bx1 + bx2) / 2
                if abs(bx_center - center_x) <= allowed_margin_x and by1 <= py1 + int(pole_height * 0.65):
                    component_boxes.append(box)

            if component_boxes:
                x1_min = min(min(box[0], px1) for box in component_boxes)
                y1_min = min(min(box[1], py1) for box in component_boxes)
                x2_max = max(max(box[2], px2) for box in component_boxes)
                y2_max = max(box[3] for box in component_boxes)

                component_span_w = max(1, x2_max - x1_min)
                max_component_pad_x = max(160, int(img_w * 0.14))
                pad_x = min(max_component_pad_x, max(60, int(component_span_w * 0.12), int(pole_width * 0.8)))
                pad_y_top = min(80, max(30, int(pole_height * 0.05)))
                pad_y_bottom = min(120, max(50, int(pole_height * 0.10)))

                crop_x1 = max(0, x1_min - pad_x)
                crop_y1 = max(0, min(py1, y1_min) - pad_y_top)
                crop_x2 = min(img_w, x2_max + pad_x)
                crop_y2 = min(img_h, y2_max + pad_y_bottom)

                if (crop_x2 - crop_x1) > 20 and (crop_y2 - crop_y1) > 20:
                    print(
                        f"DEBUG: Component-guided crop generated: primary_box={primary_box}, pad_x={pad_x}, pad_y_top={pad_y_top}, pad_y_bottom={pad_y_bottom}, x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}, components={len(component_boxes)}"
                    )
                    return crop_x1, crop_y1, crop_x2, crop_y2

            top_y2 = py1 + int(pole_height * 0.55)
            max_fallback_pad_x = max(140, int(img_w * 0.12))
            pad_x = min(max_fallback_pad_x, max(80, int(pole_width * 1.0)))
            crop_x1 = max(0, px1 - pad_x)
            crop_y1 = max(0, py1 - min(50, max(20, int(pole_height * 0.03))))
            crop_x2 = min(img_w, px2 + pad_x)
            crop_y2 = min(img_h, top_y2 + min(120, max(60, int(pole_height * 0.10))))
            if (crop_x2 - crop_x1) > 20 and (crop_y2 - crop_y1) > 20:
                print(f"DEBUG: Pole fallback crop generated: primary_box={primary_box}, pad_x={pad_x}, x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}")
                return crop_x1, crop_y1, crop_x2, crop_y2

        print("DEBUG: No usable pole/components found. Falling back to full image crop.")
        return 0, 0, img_w, img_h

    # ── Internal helpers ──────────────────────────────────────

    def _categorise(
        self, cls_name, box, conf_val, angle_deg,
        insulator_boxes, pole_boxes_raw,
        crossarm_boxes, conductor_boxes,
        street_light_boxes, other_boxes, flags,
        polygon=None, is_strut=False
    ):
        """Routes a detection into the right typed list, including polygon data."""
        det = (box, conf_val, angle_deg, polygon, is_strut)
        if _match_keyword(cls_name, "insulator"):
            det = (box, conf_val, angle_deg, polygon, cls_name)
            insulator_boxes.append(det)
        elif _match_keyword(cls_name, "pole"):
            if not meets_pole_confidence(conf_val):
                return
            # GEOMETRY SHIELD: Real main poles are tall and thin.
            # (Strut poles are exempt because their lean makes their AABB wide)
            w, h = box[2] - box[0], box[3] - box[1]
            if not is_strut and h < (w * 0.5):
                # Reject purely horizontal structures like walls/buildings
                print(f"🚫 [Geometry Shield] Rejected wide detection: {cls_name} (Aspect Ratio: {h/w:.1f})")
                return
            pole_boxes_raw.append(det)
        elif _match_keyword(cls_name, "crossarm"):
            native = cls_name.lower()
            # Structural Guard: Only use geometric check if the model didn't 
            # already tell us it's a pole.
            w, h = box[2] - box[0], box[3] - box[1]
            if (w * 1.2) > h or "t_rising" in native or "t-rising" in native or "side_arm" in native or "arm" in native or "cleat" in native or "channel" in native or "clamp" in native:
                crossarm_boxes.append((box, conf_val, angle_deg, polygon, native))
            else:
                # If it's too tall for a crossarm, it's likely a pole fragment
                # that the model mislabeled as a crossarm.
                is_strut = ("strut" in native)
                if meets_pole_confidence(conf_val):
                    pole_boxes_raw.append((box, conf_val, angle_deg, polygon, is_strut))
        elif _match_keyword(cls_name, "conductor"):
            conductor_boxes.append((box, conf_val, polygon))
        elif _match_keyword(cls_name, "lamp_head"):
            street_light_boxes.append((box, conf_val, polygon))
        elif _match_keyword(cls_name, "dtr_tank"):
            flags["has_dtr"] = True
            other_boxes.append((_normalize_label(cls_name, "dtr"), box, conf_val, polygon))
        elif _match_keyword(cls_name, "broken_wire"):
            flags["has_broken_wire"] = True
            other_boxes.append((_normalize_label(cls_name, "wire_broken"), box, conf_val, polygon))
        elif _match_keyword(cls_name, "vegetation"):
            flags["has_vegetation"] = True
            other_boxes.append((_normalize_label(cls_name, "vegetation"), box, conf_val, polygon))
        else:
            # Catch-all for any other model classes not explicitly handled
            other_boxes.append((_normalize_label(cls_name, "object"), box, conf_val, polygon))

    def _infer_pole_if_missing(
        self, insulator_boxes, crossarm_boxes, img_h, img_w
    ) -> Optional[Tuple[Tuple[int, int, int, int], float, float]]:
        """
        If we see insulators/crossarms but the model missed the pole (e.g. cut off),
        infer a vertical pole box from the component alignment.
        """
        all_box_coords = []
        for item in insulator_boxes: all_box_coords.append(item[0])
        for item in crossarm_boxes:  all_box_coords.append(item[0])
        
        if not all_box_coords:
            return None
            
        # Composite bounding box of all components
        x1_min = min(b[0] for b in all_box_coords)
        y1_min = min(b[1] for b in all_box_coords)
        x2_max = max(b[2] for b in all_box_coords)
        avg_cx = (x1_min + x2_max) / 2
        
        # Create a vertical pole box (narrow and tall)
        # We extend it to the bottom of the image because it's usually a vertical support
        pole_width = max(60, int((x2_max - x1_min) * 0.4))
        inf_x1 = int(max(0, avg_cx - pole_width // 2))
        inf_x2 = int(min(img_w, avg_cx + pole_width // 2))
        inf_y1 = int(max(0, y1_min - 20))
        inf_y2 = img_h # Extend to bottom
        
        # Return as a low-confidence detection
        return ((inf_x1, inf_y1, inf_x2, inf_y2), 0.50, 90.0)

    def _calculate_max_overlap(self, box1: tuple, box2: tuple) -> float:
        """Computes Intersection-over-Minimum-Area (IoM). Better for nested boxes."""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area  = (x2 - x1) * (y2 - y1)
        box2_area  = (x4 - x3) * (y4 - y3)
        
        min_area = min(box1_area, box2_area)
        if min_area <= 0: return 0.0
        
        return inter_area / min_area

    def _nms(self, items: list, iou_threshold: float = 0.45) -> list:
        """
        Simple NMS for list of detections using IoM overlap.
        """
        if not items:
            return []
            
        # Helper to extract (box, score, class, original_item)
        def parse_item(item):
            if isinstance(item[0], str):  # e.g., other_boxes: (label, box, conf, poly)
                return item[1], item[2], item[0], item
            else:  # e.g., (box, conf, ...)
                box = item[0]
                score = item[1]
                cls = item[4] if len(item) >= 5 else None
                return box, score, cls, item

        parsed = [parse_item(item) for item in items]
        
        # Sort parsed items by score descending
        parsed_sorted = sorted(parsed, key=lambda x: x[1], reverse=True)
        keep_parsed = []
        
        while parsed_sorted:
            best = parsed_sorted.pop(0)
            keep_parsed.append(best)
            
            remaining = []
            for item in parsed_sorted:
                # Class-aware suppression (if classes are specified and different)
                if best[2] is not None and item[2] is not None and best[2] != item[2]:
                    remaining.append(item)
                    continue

                overlap = self._calculate_max_overlap(best[0], item[0])
                # Relaxed threshold for struts (allow them to be close together)
                # best[2] is is_strut for pole boxes
                is_strut = best[2] if isinstance(best[2], bool) else False
                current_threshold = 0.85 if is_strut else iou_threshold
                if overlap < current_threshold:
                    remaining.append(item)
            parsed_sorted = remaining
            
        # Reconstruct the filtered list in their original formats
        keep = [p[3] for p in keep_parsed]
        
        # 2. Vertical Merge (Fixes fragmented poles)
        # Only apply to pole_boxes_raw (which are 5-tuples and have is_strut flag at index 4)
        is_pole_list = len(keep) > 0 and len(keep[0]) == 5 and isinstance(keep[0][4], bool)
        if not is_pole_list:
            return keep
            
        merged = []
        keep = sorted(keep, key=lambda x: x[0][1]) # Sort by Y1 (top to bottom)
        
        while keep:
            curr = keep.pop(0)
            c_box, c_conf, c_angle, c_poly, c_is_strut = curr
            
            # Look for a fragment directly below this one
            found_fragment = False
            for i, next_item in enumerate(keep):
                n_box, n_conf, n_angle, n_poly, n_is_strut = next_item
                
                # Check if they are the same type and vertically close
                if c_is_strut == n_is_strut:
                    # Check X-overlap (Vertical Alignment)
                    x_overlap = min(c_box[2], n_box[2]) - max(c_box[0], n_box[0])
                    
                    # If they share > 50% width and the gap is small
                    gap = n_box[1] - c_box[3]
                    if x_overlap / max(1, (c_box[2]-c_box[0])) > 0.5 and gap < 100:
                        # MERGE!
                        new_box = (
                            min(c_box[0], n_box[0]), 
                            min(c_box[1], n_box[1]),
                            max(c_box[2], n_box[2]),
                            max(c_box[3], n_box[3])
                        )
                        # Average angle/confidence
                        avg_conf = (c_conf + n_conf) / 2
                        avg_angle = None
                        if c_angle is not None and n_angle is not None:
                            avg_angle = (c_angle + n_angle) / 2
                        elif c_angle is not None:
                            avg_angle = c_angle
                        else:
                            avg_angle = n_angle
                            
                        new_poly = c_poly + n_poly
                        
                        # Update current and remove next_item from keep
                        curr = (new_box, avg_conf, avg_angle, new_poly, c_is_strut)
                        keep.pop(i)
                        found_fragment = True
                        break
            
            merged.append(curr)
            if found_fragment:
                # Sort again to preserve top-to-bottom order for subsequent merges
                keep = sorted(keep, key=lambda x: x[0][1])
                
        return merged






