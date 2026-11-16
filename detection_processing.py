import cv2
import numpy as np
import time

from config import normalize_detection_label, meets_pole_confidence


def merge_collinear_conductors(conductors, img_w, img_h):
    if not conductors:
        return []

    import math

    def get_endpoints(poly):
        if not poly or len(poly) < 2:
            return None, None
        pts = np.array(poly, dtype=np.float32)
        if len(pts) > 24:
            sample_idx = np.linspace(0, len(pts) - 1, 24, dtype=int)
            pts = pts[sample_idx]
        max_dist = -1.0
        p1, p2 = None, None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                distance = float(np.sum((pts[i] - pts[j]) ** 2))
                if distance > max_dist:
                    max_dist = distance
                    p1, p2 = pts[i], pts[j]
        if p1 is not None:
            return tuple(int(v) for v in p1), tuple(int(v) for v in p2)
        return tuple(int(v) for v in pts[0]), tuple(int(v) for v in pts[-1])

    def point_to_line_dist(pt, lp1, lp2):
        x0, y0 = pt
        x1, y1 = lp1
        x2, y2 = lp2
        dx = x2 - x1
        dy = y2 - y1
        denom = math.sqrt(dx * dx + dy * dy)
        if denom == 0:
            return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / denom

    def should_merge(d1, d2):
        poly1 = d1.get("polygon")
        poly2 = d2.get("polygon")
        if not poly1 or not poly2:
            return False
        e11, e12 = get_endpoints(poly1)
        e21, e22 = get_endpoints(poly2)
        if not e11 or not e21:
            return False

        v1 = np.array(e12) - np.array(e11)
        v2 = np.array(e22) - np.array(e21)
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 == 0 or len2 == 0:
            return False

        u1 = v1 / len1
        u2 = v2 / len2
        if abs(np.dot(u1, u2)) < 0.95:
            return False

        distances = [
            point_to_line_dist(e21, e11, e12),
            point_to_line_dist(e22, e11, e12),
            point_to_line_dist(e11, e21, e22),
            point_to_line_dist(e12, e21, e22),
        ]
        if max(distances) > 30.0:
            return False

        gap = min(
            np.linalg.norm(np.array(e11) - np.array(e21)),
            np.linalg.norm(np.array(e11) - np.array(e22)),
            np.linalg.norm(np.array(e12) - np.array(e21)),
            np.linalg.norm(np.array(e12) - np.array(e22)),
        )
        return gap <= max(400.0, img_w * 0.35)

    def merge_two(d1, d2):
        combined_pts = np.array(d1.get("polygon", []) + d2.get("polygon", []), dtype=np.int32)
        hull = cv2.convexHull(combined_pts)
        merged_poly = [[int(pt[0][0]), int(pt[0][1])] for pt in hull]
        b1 = d1.get("bbox", [0, 0, 0, 0])
        b2 = d2.get("bbox", [0, 0, 0, 0])
        merged_bbox = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
        return {
            "label": normalize_detection_label("conductor"),
            "confidence": max(d1.get("confidence", 0.90), d2.get("confidence", 0.90)),
            "bbox": merged_bbox,
            "polygon": merged_poly,
            "source": "models/cable_best.pt",
            "thickness": round((d1.get("thickness", 0.0) + d2.get("thickness", 0.0)) / 2.0, 1),
        }

    current = list(conductors)
    if len(current) > 24:
        print(f"[Conductor Merge] Skipping expensive merge for {len(current)} candidates; returning capped raw conductors.")
        current = sorted(current, key=lambda item: (item.get('bbox', [0, 0, 0, 0])[2] - item.get('bbox', [0, 0, 0, 0])[0]) * (item.get('bbox', [0, 0, 0, 0])[3] - item.get('bbox', [0, 0, 0, 0])[1]), reverse=True)
        return current[:24]

    merge_start = time.perf_counter()
    merged_any = True
    passes = 0
    while merged_any and passes < 24:
        passes += 1
        merged_any = False
        i = 0
        while i < len(current):
            j = i + 1
            while j < len(current):
                if should_merge(current[i], current[j]):
                    current[i] = merge_two(current[i], current[j])
                    current.pop(j)
                    merged_any = True
                    break
                j += 1
            if merged_any:
                break
            i += 1
    print(f"[Stage Timing] conductor_merge={time.perf_counter() - merge_start:.3f}s, remaining={len(current)}, passes={passes}")
    return current


def collect_model_detections(pipe_res, width, height):
    hardware_mask = np.zeros((height, width), dtype=np.uint8)
    final_detections = []
    main_pole_boxes = []

    primary_pole = pipe_res.pole_orientation
    if primary_pole is not None and not meets_pole_confidence(primary_pole.detection_conf):
        primary_pole = None
    has_primary = primary_pole is not None
    if has_primary:
        pp_x1, _, pp_x2, _ = primary_pole.box
        margin_x = max(int(width * 0.18), (pp_x2 - pp_x1) * 4)
        allowed_x1 = pp_x1 - margin_x
        allowed_x2 = pp_x2 + margin_x
        print(f"[HUD Filter] Primary pole center bounds: [{allowed_x1}, {allowed_x2}]")
    else:
        allowed_x1 = allowed_x2 = None

    for ins in pipe_res.insulators:
        x1, y1, x2, y2 = [int(v) for v in ins.box]
        center_x = (x1 + x2) / 2
        if has_primary and not (allowed_x1 <= center_x <= allowed_x2):
            print(f"[HUD Filter] Skipped background insulator at cx={center_x}")
            continue
        cv2.rectangle(hardware_mask, (max(0, x1 - 5), max(0, y1 - 5)), (min(width, x2 + 5), min(height, y2 + 5)), 255, -1)
        final_detections.append({
            "label": normalize_detection_label(ins.type_heuristic or ins.type_final or "insulator"),
            "confidence": float(ins.detection_conf),
            "bbox": [int(x) for x in ins.box],
            "polygon": getattr(ins, "polygon", None),
            "source": "models/insulator_new.pt",
            "details": {"raw_label": normalize_detection_label(ins.type_heuristic or ins.type_final or "insulator"), "type": normalize_detection_label(ins.type_heuristic or ins.type_final or "insulator")},
        })

    for crossarm in pipe_res.crossarms:
        x1, y1, x2, y2 = [int(v) for v in crossarm.box]
        center_x = (x1 + x2) / 2
        if has_primary and not (allowed_x1 <= center_x <= allowed_x2):
            print(f"[HUD Filter] Skipped background crossarm at cx={center_x}")
            continue
        cv2.rectangle(hardware_mask, (max(0, x1 - 5), max(0, y1 - 5)), (min(width, x2 + 5), min(height, y2 + 5)), 255, -1)
        final_detections.append({
            "label": normalize_detection_label(crossarm.shape or "crossarm"),
            "confidence": float(crossarm.detection_conf) * 0.75,
            "bbox": [int(x) for x in crossarm.box],
            "polygon": getattr(crossarm, "polygon", None),
            "source": "models/channel_12class_v2.pt",
            "details": {"raw_label": normalize_detection_label(crossarm.shape or "crossarm")},
        })

    for pole in pipe_res.all_poles:
        if not meets_pole_confidence(pole.detection_conf):
            continue
        final_detections.append({
            "label": normalize_detection_label(pole.pole_type or "pole"),
            "confidence": float(pole.detection_conf),
            "bbox": [int(x) for x in pole.box],
            "polygon": getattr(pole, "polygon", None),
            "source": "models/pole_best6.pt",
            "details": {"type": pole.pole_type, "lean": round(float(pole.lean_angle_deg), 1)},
        })
        if pole.pole_type != "strut_pole":
            main_pole_boxes.append([int(v) for v in pole.box])

    for box, conf, poly in pipe_res.street_lights:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(hardware_mask, (max(0, x1 - 5), max(0, y1 - 5)), (min(width, x2 + 5), min(height, y2 + 5)), 255, -1)
        final_detections.append({
            "label": normalize_detection_label("street_light"),
            "confidence": float(conf) * 0.75,
            "bbox": [int(x) for x in box],
            "polygon": poly,
            "source": "models/channel_12class_v2.pt",
            "details": {"type": "Standard Lamp"},
        })

    for label, box, conf, poly in pipe_res.others:
        if (box[2] - box[0]) > 100 or (box[3] - box[1]) > 100:
            cv2.rectangle(hardware_mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 255, -1)
        final_detections.append({
            "label": normalize_detection_label(label),
            "confidence": float(conf),
            "bbox": [int(x) for x in box],
            "polygon": poly,
            "details": {"source": "AI Inference"},
        })

    # ── Tapping_Arm → 3_wire_crossarm / 5_wire_crossarm spatial mapping ──────
    # Strategy: dual-check using OVERLAP + PROXIMITY CLUSTER
    #
    #  In real images, insulators are mounted at the TIP of the cross arm —
    #  they often don't overlap the arm bbox, they extend beyond it.
    #  So we expand the tapping_arm bbox by a proximity margin to form a
    #  "neighborhood zone" covering the full arm extent and check if insulator
    #  centroids fall within that zone.
    #
    #  Step 1: If neither ht_pin/ht_disc NOR lt_pin/shackle are detected → warn + skip.
    #  Step 2: For each tapping_arm, build its neighborhood zone:
    #          • Check direct overlap (10% IoMin threshold)
    #          • Check proximity: insulator centroid inside expanded arm bbox
    #            (arm expanded by arm_width*1.2 horizontally, arm_height*0.8 vertically)
    #          Decide:
    #          • ht_pin or ht_disc in zone  → keep as tapping_arm
    #          • lt_pin or shackle in zone  → remap (count ≥ 4 → 5_wire, else 3_wire)
    #          • nothing in zone            → keep as tapping_arm

    def _boxes_overlap(a, b, threshold=0.10):
        """True if intersection/min-area >= threshold (direct overlap)."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return (inter / min(area_a, area_b)) >= threshold

    def _centroid_in_zone(arm_bbox, ins_bbox, h_expand=1.2, v_expand=0.8):
        """True if insulator centroid falls within the expanded arm neighborhood zone."""
        ax1, ay1, ax2, ay2 = arm_bbox
        aw = ax2 - ax1
        ah = ay2 - ay1
        # Expand arm bbox — cross arms extend horizontally, so wider margin on x
        ex1 = ax1 - aw * h_expand
        ey1 = ay1 - ah * v_expand
        ex2 = ax2 + aw * h_expand
        ey2 = ay2 + ah * v_expand
        # Get insulator centroid
        ix1, iy1, ix2, iy2 = ins_bbox
        cx = (ix1 + ix2) / 2
        cy = (iy1 + iy2) / 2
        return ex1 <= cx <= ex2 and ey1 <= cy <= ey2

    def _in_arm_cluster(arm_bbox, ins_bbox):
        """Combined check: direct overlap OR centroid falls in proximity zone."""
        if len(ins_bbox) != 4:
            return False
        return _boxes_overlap(arm_bbox, ins_bbox) or _centroid_in_zone(arm_bbox, ins_bbox)

    ht_insulators = [d for d in final_detections
                     if any(k in str(d.get("label", "")).lower() for k in ["ht_pin", "ht_disc"])]
    lt_insulators = [d for d in final_detections
                     if "lt_pin" in str(d.get("label", "")).lower()
                     or "shackle" in str(d.get("label", "")).lower()]

    has_ht = len(ht_insulators) > 0
    has_lt = len(lt_insulators) > 0

    # Step 1: warn if neither HT nor LT insulators are detected at all
    if not has_ht and not has_lt:
        print("[CrossarmMap] WARNING: HT pins and LT pins are not getting detected — skipping tapping_arm remapping")
    else:
        # Step 2: check each tapping_arm using overlap + proximity cluster
        for det in final_detections:
            if str(det.get("label", "")).lower() != "tapping_arm":
                continue

            ta_bbox = det.get("bbox", [])
            if not ta_bbox or len(ta_bbox) != 4:
                continue

            # Check HT cluster (overlap OR proximity)
            ht_in_zone = any(_in_arm_cluster(ta_bbox, ht.get("bbox", [])) for ht in ht_insulators)
            if ht_in_zone:
                print(f"[CrossarmMap] tapping_arm kept — ht_pin/ht_disc in arm cluster zone")
                continue

            # Check LT cluster (overlap OR proximity)
            lt_in_zone = [lt for lt in lt_insulators if _in_arm_cluster(ta_bbox, lt.get("bbox", []))]

            if lt_in_zone:
                lt_count = len(lt_in_zone)
                det["label"] = "5_wire_crossarm" if lt_count >= 4 else "3_wire_crossarm"
                print(f"[CrossarmMap] tapping_arm → {det['label']} (lt_in_cluster={lt_count})")
            else:
                # Nothing in cluster zone → don't apply logic
                print(f"[CrossarmMap] tapping_arm kept — no ht or lt in arm cluster zone")

    return final_detections, hardware_mask, main_pole_boxes


def extract_conductor_detections(mask_closed, hardware_mask, main_pole_boxes, dist, skel, width, height):
    extract_start = time.perf_counter()
    mask_final = cv2.bitwise_and(mask_closed, cv2.bitwise_not(hardware_mask))
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 80:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:80]
        print(f"[Conductor Filter] Contours capped to 80 for runtime safety.")
    raw_conductors = []

    contour_stage_start = time.perf_counter()
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if cw + ch < 60 or area < 60:
            continue

        aspect_ratio = max(cw, ch) / max(1, min(cw, ch))
        solidity = area / (cw * ch)
        if cw > 150 and ch > 150 and solidity > 0.5 and aspect_ratio < 1.8:
            continue

        overlaps_main_pole = False
        for px1, py1, px2, py2 in main_pole_boxes:
            threshold_y = py1 + max(150, int((py2 - py1) * 0.45))
            if cx <= px2 and (cx + cw) >= px1 and cy <= threshold_y:
                overlaps_main_pole = True
                break
        if not overlaps_main_pole:
            continue

        approx = cv2.approxPolyDP(contour, 3.0, True)
        if len(approx) > 32:
            approx = approx[:32]
        polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        local_skel = skel & (contour_mask > 0)
        local_thickness = dist[local_skel > 0] * 2
        avg_thick = float(np.mean(local_thickness)) if len(local_thickness) > 0 else 0.0
        if avg_thick > 100:
            continue

        raw_conductors.append({
            "label": normalize_detection_label("conductor"),
            "confidence": 0.70,
            "bbox": [cx, cy, cx + cw, cy + ch],
            "polygon": polygon,
            "source": "models/cable_best.pt",
            "thickness": round(avg_thick, 1),
        })

    print(f"[Stage Timing] conductor_contours={time.perf_counter() - contour_stage_start:.3f}s, raw_count={len(raw_conductors)}")
    merged = merge_collinear_conductors(raw_conductors, width, height)
    print(f"[Stage Timing] conductor_total={time.perf_counter() - extract_start:.3f}s, merged_count={len(merged)}")
    return merged


def _infer_voltage_from_detections(pipe_res):
    """Infer voltage level from detected insulator types and class labels."""
    ht_labels = {"ht_pin", "ht_disc", "11kv", "33kv", "v_cross", "v_cross_arm", "t_rising", "t_raising"}
    lt_labels = {"lt_pin", "shackle", "shackle_insulator", "3_wire", "5_wire", "3_wire_crossarm", "5_wire_crossarm"}
    dtr_labels = {"dtr", "dtr_tank", "transformer"}

    has_ht = False
    has_lt = False
    has_dtr = False
    voltage_sources = []

    # Check insulator types
    for ins in pipe_res.insulators:
        label = str(getattr(ins, "type_heuristic", "") or getattr(ins, "type_final", "") or "").lower()
        if any(kw in label for kw in ht_labels):
            has_ht = True
            voltage_sources.append(f"insulator:{label}")
        elif any(kw in label for kw in lt_labels):
            has_lt = True
            voltage_sources.append(f"insulator:{label}")

    # Check crossarms
    for arm in pipe_res.crossarms:
        label = str(getattr(arm, "shape", "") or "").lower()
        if any(kw in label for kw in ht_labels):
            has_ht = True
            voltage_sources.append(f"crossarm:{label}")
        elif any(kw in label for kw in lt_labels):
            has_lt = True
            voltage_sources.append(f"crossarm:{label}")

    # Check others for DTR
    for item in pipe_res.others:
        label = str(item[0] if isinstance(item, (list, tuple)) else "").lower()
        if any(kw in label for kw in dtr_labels):
            has_dtr = True
            voltage_sources.append(f"other:{label}")

    # Determine voltage
    if has_dtr and has_ht:
        voltage = "11KV HT (with DTR)"
    elif has_dtr:
        voltage = "Distribution (DTR)"
    elif has_ht and has_lt:
        voltage = "11KV HT / LT Mixed"
    elif has_ht:
        voltage = "11KV HT"
    elif has_lt:
        voltage = "LT (Low Tension)"
    else:
        # Fallback: check final_class name
        fc = str(pipe_res.final_class or "").lower()
        if "11kv" in fc or "ht" in fc:
            voltage = "11KV HT"
        elif "lt" in fc:
            voltage = "LT (Low Tension)"
        else:
            voltage = "Unknown (Insufficient Data)"

    return voltage, voltage_sources


def build_master_data(pipe_res):
    voltage, voltage_sources = _infer_voltage_from_detections(pipe_res)
    primary = pipe_res.pole_orientation
    if primary is not None and not meets_pole_confidence(primary.detection_conf):
        primary = None
    return {
        "final_class": pipe_res.final_class,
        "voltage": voltage,
        "voltage_sources": voltage_sources,
        "pole_id": pipe_res.pole_id if primary else "Not Found",
        "reason": pipe_res.reason,
        "confidence": pipe_res.confidence,
        "pole_lean_angle": primary.lean_angle_deg if primary else 0.0,
        "pole_type": primary.pole_type if primary else "none",
        "pole_status": "raw_mode" if primary else "not_detected",
        "model_summary": {
            "structural": "models/channel_12class_v2.pt",
            "insulator": "models/insulator_new.pt",
            "segmentation": "models/cable_best.pt",
            "mode": "raw_predictions",
        },
    }


def build_survey_questionnaire(pipe_res, conductors):
    return {
        "raw_mode": "Yes",
        "dominant_label": pipe_res.final_class,
        "pole_count": sum(1 for pole in pipe_res.all_poles if meets_pole_confidence(pole.detection_conf)),
        "strut_pole_count": sum(1 for pole in pipe_res.all_poles if pole.pole_type == "strut_pole" and meets_pole_confidence(pole.detection_conf)),
        "conductor_count": len(conductors),
        "vegetation_detected": "Yes" if pipe_res.flags.get("has_vegetation") else "No",
    }
