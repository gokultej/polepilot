import io
import json
import math
from datetime import datetime

import cv2
import numpy as np

from image_processing import process_image_file
from routes.tree_routes import (
    _azure_runtime_configs,
    _extract_token_usage,
    _image_bytes_to_data_url,
    _run_azure_tree_completion,
    _safe_json_loads,
)
from services.distance_estimation_service import (
    bbox_to_bbox_distance_px,
    build_distance_metrics_for_tree,
    calculate_bbox_height_px,
    estimate_distance_using_pole_height,
    normalized_bbox_to_px,
)


def infer_season(capture_date, latitude=None):
    if isinstance(capture_date, str):
        try:
            capture_date = datetime.fromisoformat(capture_date.replace('Z', '+00:00'))
        except Exception:
            capture_date = datetime.now()
    elif capture_date is None:
        capture_date = datetime.now()

    month = capture_date.month
    northern = True
    try:
        if latitude is not None and float(latitude) < 0:
            northern = False
    except Exception:
        northern = True

    if northern:
        if month in (12, 1, 2):
            return 'winter'
        if month in (3, 4, 5):
            return 'summer'
        if month in (6, 7, 8, 9):
            return 'monsoon'
        return 'post_monsoon'

    if month in (6, 7, 8):
        return 'winter'
    if month in (9, 10, 11):
        return 'spring'
    if month in (12, 1, 2):
        return 'summer'
    return 'autumn'


def _decode_image(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Unable to decode image bytes')
    return image


def _bbox_from_detection(det):
    bbox = det.get('bbox')
    if isinstance(bbox, list) and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
        return [float(x1), float(y1), float(x2), float(y2)]
    polygon = det.get('polygon')
    if isinstance(polygon, list) and polygon:
        xs = [float(pt[0]) for pt in polygon]
        ys = [float(pt[1]) for pt in polygon]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _normalize_bbox(box, width, height):
    x1, y1, x2, y2 = box
    return {
        'x_min': round(max(0.0, min(1.0, x1 / max(1, width))), 6),
        'y_min': round(max(0.0, min(1.0, y1 / max(1, height))), 6),
        'x_max': round(max(0.0, min(1.0, x2 / max(1, width))), 6),
        'y_max': round(max(0.0, min(1.0, y2 / max(1, height))), 6),
    }


def _clip_box(box, width, height):
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(float(width), float(x1))),
        max(0.0, min(float(height), float(y1))),
        max(0.0, min(float(width), float(x2))),
        max(0.0, min(float(height), float(y2))),
    ]


def _normalize_line(line, width, height):
    if not isinstance(line, dict):
        return None
    try:
        x1 = float(line.get('x1', 0.0))
        y1 = float(line.get('y1', 0.0))
        x2 = float(line.get('x2', 0.0))
        y2 = float(line.get('y2', 0.0))
    except (TypeError, ValueError):
        return None
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        return {
            'x1': round(max(0.0, min(1.0, x1 / max(1, width))), 6),
            'y1': round(max(0.0, min(1.0, y1 / max(1, height))), 6),
            'x2': round(max(0.0, min(1.0, x2 / max(1, width))), 6),
            'y2': round(max(0.0, min(1.0, y2 / max(1, height))), 6),
        }
    return {
        'x1': round(max(0.0, min(1.0, x1)), 6),
        'y1': round(max(0.0, min(1.0, y1)), 6),
        'x2': round(max(0.0, min(1.0, x2)), 6),
        'y2': round(max(0.0, min(1.0, y2)), 6),
    }


def _wire_lines_from_detections(detections, width, height):
    lines = []
    for det in detections:
        label = str(det.get('label', '')).upper()
        if 'CONDUCTOR' not in label and 'WIRE' not in label:
            continue
        box = _bbox_from_detection(det)
        if not box:
            continue
        x1, y1, x2, y2 = box
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        if box_w >= box_h:
            line = {'x1': x1 / width, 'y1': ((y1 + y2) / 2.0) / height, 'x2': x2 / width, 'y2': ((y1 + y2) / 2.0) / height}
        else:
            line = {'x1': ((x1 + x2) / 2.0) / width, 'y1': y1 / height, 'x2': ((x1 + x2) / 2.0) / width, 'y2': y2 / height}
        lines.append(_normalize_line(line, width, height))
    return [line for line in lines if line]


def _build_measurement_bundle(trees, pole_box, wire_lines, width, height, image_quality=None):
    pole_bbox_normalized = _normalize_bbox(pole_box, width, height) if pole_box else None
    pole_bbox_px = normalized_bbox_to_px(pole_bbox_normalized, width, height) if pole_bbox_normalized else None
    pole_height_px = calculate_bbox_height_px(pole_bbox_px) if pole_bbox_px else 0.0
    meters_per_pixel = round(9.0 / pole_height_px, 5) if pole_height_px > 0 else None
    pole_clipped = False
    if pole_bbox_px:
        pole_clipped = pole_bbox_px['y1'] <= 2 or pole_bbox_px['y2'] >= height - 2

    can_estimate = bool(pole_bbox_px and pole_height_px >= height * 0.25 and wire_lines and not pole_clipped)
    reason = ''
    if not pole_bbox_px:
        reason = 'Pole is not visible.'
    elif pole_clipped:
        reason = 'Full pole height is not visible.'
    elif pole_height_px < height * 0.25:
        reason = 'Pole bounding box is too small for reliable scale.'
    elif not wire_lines:
        reason = 'Wire geometry is not visible.'

    if image_quality and not image_quality.get('is_usable', True) and not reason:
        reason = 'Image quality is not reliable enough for distance estimation.'

    for tree in trees:
        metrics = build_distance_metrics_for_tree(tree.get('bbox_normalized'), wire_lines, pole_bbox_normalized, width, height, real_pole_height_m=9.0)
        if not can_estimate:
            metrics['meters_per_pixel'] = None
            metrics['estimated_wire_clearance_m'] = None if not metrics.get('is_touching_wire') else 0.0
            metrics['estimated_pole_distance_m'] = None
            metrics['distance_confidence'] = 'low'
            metrics['distance_note'] = f"Distance could not be estimated because {reason.lower()} Risk is based on visual proximity only."
            metrics['clearance_label'] = 'touching' if metrics.get('is_touching_wire') else 'unknown'
        tree['distance_metrics'] = metrics
        recomputed_risk = metrics.get('recomputed_risk_level')
        if recomputed_risk in {'critical', 'high', 'medium', 'low', 'no_risk'} and recomputed_risk != 'unknown':
            risk_order = {'no_risk': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            current = tree.get('risk_level', 'no_risk')
            if risk_order.get(recomputed_risk, 0) > risk_order.get(current, 0):
                tree['risk_level'] = recomputed_risk
                if metrics.get('estimated_wire_clearance_m') is not None:
                    tree['risk_reason'] = f"Estimated wire clearance is about {metrics['estimated_wire_clearance_m']} m from the assumed 9 m pole scale."
                elif metrics.get('is_touching_wire'):
                    tree['risk_reason'] = 'Vegetation appears to be touching the conductor.'
            tree['is_affecting_pole'] = tree['risk_level'] in {'medium', 'high', 'critical'}
            tree['species_detection_required'] = tree['risk_level'] in {'medium', 'high', 'critical'} or bool(tree.get('species_detection_required'))

    tree_to_tree = []
    if pole_bbox_px and pole_height_px > 0:
        for index, tree in enumerate(trees):
            bbox_a = normalized_bbox_to_px(tree.get('bbox_normalized'), width, height)
            for other in trees[index + 1:]:
                bbox_b = normalized_bbox_to_px(other.get('bbox_normalized'), width, height)
                px_distance = bbox_to_bbox_distance_px(bbox_a, bbox_b)
                if px_distance is None:
                    continue
                estimate = estimate_distance_using_pole_height(pole_height_px, px_distance, real_pole_height_m=9.0)
                tree_to_tree.append({
                    'from_tree': tree.get('tree_id'),
                    'to_tree': other.get('tree_id'),
                    'distance_px': px_distance,
                    'estimated_distance_m': estimate.get('estimated_distance_m') if can_estimate else None,
                    'distance_note': 'Estimated using assumed 9 m pole height.' if can_estimate else 'Distance estimate unavailable because pole scale is incomplete.',
                })

    return {
        'measurement_config': {
            'assumed_pole_height_m': 9.0,
            'measurement_type': 'single_image_estimate',
            'is_survey_grade': False,
            'note': 'Distances are estimated using assumed 9 m pole height and image pixel scale.',
        },
        'pole_measurement': {
            'pole_bbox': pole_bbox_normalized or {},
            'pole_height_px': round(pole_height_px, 2) if pole_height_px else 0.0,
            'meters_per_pixel': meters_per_pixel if can_estimate else None,
            'can_estimate_distance': can_estimate,
            'reason': reason,
        },
        'wire_lines': wire_lines,
        'tree_to_tree_distances': tree_to_tree,
    }


def _box_intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _box_distance(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


def _merge_boxes(boxes, merge_gap=40):
    merged = []
    for box in boxes:
        current = list(box)
        changed = True
        while changed:
            changed = False
            next_round = []
            for existing in merged:
                inflated_existing = [existing[0] - merge_gap, existing[1] - merge_gap, existing[2] + merge_gap, existing[3] + merge_gap]
                if _box_intersection(current, inflated_existing) > 0:
                    current = [
                        min(current[0], existing[0]),
                        min(current[1], existing[1]),
                        max(current[2], existing[2]),
                        max(current[3], existing[3]),
                    ]
                    changed = True
                else:
                    next_round.append(existing)
            merged = next_round
        merged.append(current)
    return merged


def _detect_vegetation_regions(image, detections):
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([25, 30, 20]), np.array([95, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([10, 20, 20]), np.array([35, 255, 180]))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    boxes = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = max(1500, int(width * height * 0.01))
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < min_area:
            continue
        if h < 30 or w < 30:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])

    vegetation_dets = []
    for det in detections:
        label = str(det.get('label', '')).upper()
        if 'VEGETATION' in label or 'TREE' in label or 'BRANCH' in label:
            box = _bbox_from_detection(det)
            if box:
                vegetation_dets.append(box)

    boxes.extend(vegetation_dets)
    if not boxes:
        boxes = [[0.0, 0.0, float(width), float(height)]]

    merged = _merge_boxes(boxes, merge_gap=max(25, int(width * 0.02)))
    clipped = []
    for box in merged:
        clipped_box = _clip_box(box, width, height)
        if clipped_box[2] - clipped_box[0] >= 40 and clipped_box[3] - clipped_box[1] >= 40:
            clipped.append(clipped_box)
    return clipped


def _detected_object_summary(detections):
    pole_boxes = []
    wire_boxes = []
    for det in detections:
        label = str(det.get('label', '')).upper()
        box = _bbox_from_detection(det)
        if not box:
            continue
        if 'POLE' in label:
            pole_boxes.append(box)
        if 'CONDUCTOR' in label or 'WIRE' in label:
            wire_boxes.append(box)
    return pole_boxes, wire_boxes


def _recompute_overall_risk(trees):
    overall_risk = 'no_risk'
    order = ['unknown', 'no_risk', 'low', 'medium', 'high', 'critical']
    for tree in trees:
        level = tree.get('risk_level', 'no_risk')
        if order.index(level) > order.index(overall_risk):
            overall_risk = level
    return overall_risk


def _relative_position(tree_box, pole_box):
    if not pole_box:
        return 'unknown'
    tree_cx = (tree_box[0] + tree_box[2]) / 2.0
    pole_cx = (pole_box[0] + pole_box[2]) / 2.0
    margin = max(15.0, (pole_box[2] - pole_box[0]) * 0.5)
    if tree_cx < pole_cx - margin:
        return 'left'
    if tree_cx > pole_cx + margin:
        return 'right'
    return 'front'


def _risk_from_geometry(tree_box, pole_boxes, wire_boxes, image_diag):
    min_pole_dist = min((_box_distance(tree_box, pole) for pole in pole_boxes), default=image_diag)
    min_wire_dist = min((_box_distance(tree_box, wire) for wire in wire_boxes), default=image_diag)
    touching_wire = any(_box_intersection(tree_box, wire) > 0 for wire in wire_boxes)
    touching_pole = any(_box_intersection(tree_box, pole) > 0 for pole in pole_boxes)
    risk_distance = min(min_pole_dist, min_wire_dist)

    if touching_wire or touching_pole:
        return 'critical', 'Vegetation appears to be touching the pole or conductor', True
    if risk_distance <= image_diag * 0.04:
        return 'high', 'Vegetation appears very close to the pole or conductor', True
    if risk_distance <= image_diag * 0.10:
        return 'medium', 'Vegetation appears to be growing toward the pole or conductor', True
    if risk_distance <= image_diag * 0.18:
        return 'low', 'Vegetation is visible near the pole corridor but not immediately threatening', False
    return 'no_risk', 'Vegetation appears away from the pole and conductor corridor', False


def _recommended_action(risk_level):
    return {
        'critical': 'urgent_action',
        'high': 'trim_required',
        'medium': 'supervisor_review',
        'low': 'monitor',
        'no_risk': 'no_action',
    }.get(risk_level, 'monitor')


def _empty_plantnet_result():
    return {
        'best_match': '',
        'scientific_name': '',
        'common_name': '',
        'confidence': 0.0,
        'top_predictions': [],
    }


def _build_summary(trees):
    if not trees:
        return 'No vegetation regions were isolated from the current pole context image.'
    highest = max((tree['risk_level'] for tree in trees), key=lambda value: ['unknown', 'no_risk','low','medium','high','critical'].index(value))
    risky = [tree for tree in trees if tree['risk_level'] in {'medium', 'high', 'critical'}]
    if risky:
        measured = [tree for tree in risky if tree.get('distance_metrics', {}).get('estimated_wire_clearance_m') is not None]
        if measured:
            nearest = min(tree['distance_metrics']['estimated_wire_clearance_m'] for tree in measured)
            return f'{len(risky)} vegetation region(s) require follow-up. Highest observed risk is {highest}. Nearest estimated wire clearance is about {nearest} m.'
        return f'{len(risky)} vegetation region(s) require follow-up. Highest observed risk is {highest}.'
    return 'Vegetation is visible, but no region appears to be an active threat from this image.'


def _normalize_tree_item(raw, width, height, idx):
    bbox = raw.get('bbox_normalized') or raw.get('bbox') or {}
    if isinstance(bbox, list) and len(bbox) >= 4:
        bbox = {
            'x_min': bbox[0],
            'y_min': bbox[1],
            'x_max': bbox[2],
            'y_max': bbox[3],
        }
    x_min = max(0.0, min(1.0, float(bbox.get('x_min', 0.0))))
    y_min = max(0.0, min(1.0, float(bbox.get('y_min', 0.0))))
    x_max = max(0.0, min(1.0, float(bbox.get('x_max', 1.0))))
    y_max = max(0.0, min(1.0, float(bbox.get('y_max', 1.0))))
    if x_max <= x_min:
        x_min, x_max = 0.0, 1.0
    if y_max <= y_min:
        y_min, y_max = 0.0, 1.0
    risk_level = str(raw.get('risk_level') or 'medium').lower()
    if risk_level not in {'no_risk', 'low', 'medium', 'high', 'critical'}:
        risk_level = 'medium'
    return {
        'tree_id': raw.get('tree_id') or f'tree_{idx}',
        'position': str(raw.get('position') or 'unknown').lower(),
        'bbox_normalized': {
            'x_min': round(x_min, 6),
            'y_min': round(y_min, 6),
            'x_max': round(x_max, 6),
            'y_max': round(y_max, 6),
        },
        'risk_level': risk_level,
        'risk_reason': str(raw.get('risk_reason') or 'Risk estimated from visible tree and wire proximity.')[:240],
        'is_affecting_pole': bool(raw.get('is_affecting_pole', risk_level in {'medium', 'high', 'critical'})),
        'species_detection_required': bool(raw.get('species_detection_required', risk_level in {'medium', 'high', 'critical'})),
        'plantnet_result': _empty_plantnet_result(),
        'recommended_action': str(raw.get('recommended_action') or _recommended_action(risk_level)),
    }


def _fallback_context_analysis(image_bytes, latitude=None, longitude=None, capture_date=None, season=None):
    image = _decode_image(image_bytes)
    height, width = image.shape[:2]
    diag = math.hypot(width, height)
    try:
        inference = process_image_file(io.BytesIO(image_bytes))
        detections = inference.get('detections') or []
    except Exception:
        detections = []

    pole_boxes, wire_boxes = _detected_object_summary(detections)
    vegetation_boxes = _detect_vegetation_regions(image, detections)
    primary_pole = pole_boxes[0] if pole_boxes else None
    wire_lines = _wire_lines_from_detections(detections, width, height)

    trees = []
    for idx, box in enumerate(vegetation_boxes, start=1):
        risk_level, risk_reason, affecting = _risk_from_geometry(box, pole_boxes, wire_boxes, diag)
        trees.append({
            'tree_id': f'tree_{idx}',
            'position': _relative_position(box, primary_pole),
            'bbox_normalized': _normalize_bbox(box, width, height),
            'risk_level': risk_level,
            'risk_reason': risk_reason,
            'is_affecting_pole': affecting,
            'species_detection_required': risk_level in {'medium', 'high', 'critical'},
            'plantnet_result': _empty_plantnet_result(),
            'recommended_action': _recommended_action(risk_level),
        })

    usable = bool(pole_boxes and wire_boxes)
    issues = []
    extra_instruction = ''
    if not pole_boxes:
        issues.append('pole_not_visible')
        extra_instruction = 'Capture one clearer pole + wire context photo from a safe distance.'
    elif not wire_boxes:
        issues.append('wire_not_visible')
        extra_instruction = 'Capture one clearer pole + wire context photo from a safe distance.'

    image_quality = {
        'is_usable': usable,
        'issues': issues,
        'recommendation': extra_instruction or 'Image is usable for a first-pass vegetation review.',
    }
    measurement_bundle = _build_measurement_bundle(trees, primary_pole, wire_lines, width, height, image_quality=image_quality)
    overall_risk = _recompute_overall_risk(trees)

    return {
        'pole_id': '',
        'input': {
            'latitude': float(latitude) if latitude not in (None, '') else 0.0,
            'longitude': float(longitude) if longitude not in (None, '') else 0.0,
            'capture_date': capture_date or datetime.now().isoformat(),
            'season': season or infer_season(capture_date, latitude),
        },
        'measurement_config': measurement_bundle['measurement_config'],
        'image_quality': image_quality,
        'detected_objects': {
            'pole_visible': bool(pole_boxes),
            'wire_visible': bool(wire_boxes),
            'vegetation_count': len(trees),
        },
        'pole_measurement': measurement_bundle['pole_measurement'],
        'wire_lines': measurement_bundle['wire_lines'],
        'trees': trees,
        'tree_to_tree_distances': measurement_bundle['tree_to_tree_distances'],
        'overall_risk': overall_risk,
        'summary': _build_summary(trees),
        'ask_extra_photo': bool(extra_instruction),
        'extra_photo_instruction': extra_instruction,
        'safety_instructions': [
            'Do not touch wires, poles, or tree branches.',
            'Do not stand under wires.',
            'Do not climb the pole or the tree.',
            'Capture photos from a safe distance only.',
        ],
        'warnings': ['Fallback vegetation analysis was used because Azure context analysis was unavailable.'],
    }


def analyze_pole_context_with_azure(image_bytes, lat=None, lon=None, date=None, season=None):
    fallback = _fallback_context_analysis(image_bytes, latitude=lat, longitude=lon, capture_date=date, season=season)
    fallback['token_usage'] = {
        'provider': 'azure_openai',
        'status': 'not_used',
        'image_count': 1,
    }
    configs = _azure_runtime_configs()
    if not configs:
        return fallback

    prompt = f"""
You are analyzing one pole vegetation inspection image.

Return valid JSON only.

Task:
- Detect all visible vegetation or tree regions near the pole and wires.
- Label them as tree_1, tree_2, tree_3...
- Estimate relative position: left, right, front, behind, or unknown.
- Judge risk from spatial relation to the pole and conductor, not from species.
- Risk levels: no_risk, low, medium, high, critical.
- Provide bounding boxes normalized between 0 and 1 as x_min, y_min, x_max, y_max.
- Provide wire line coordinates normalized between 0 and 1 as x1, y1, x2, y2 when visible.
- Include short risk_reason and recommended_action.
- Ask for only one extra photo if pole/wire is unclear or the risky tree is occluded.

Context:
latitude={lat}
longitude={lon}
capture_date={date}
season={season}

Return this JSON schema:
{{
  "image_quality": {{"is_usable": true, "issues": [], "recommendation": ""}},
  "detected_objects": {{"pole_visible": true, "wire_visible": true, "vegetation_count": 0}},
  "wire_lines": [{{"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}}],
  "trees": [
    {{
      "tree_id": "tree_1",
      "position": "left",
      "bbox_normalized": {{"x_min": 0.0, "y_min": 0.0, "x_max": 0.0, "y_max": 0.0}},
      "risk_level": "medium",
      "risk_reason": "",
      "is_affecting_pole": true,
      "species_detection_required": true,
      "recommended_action": "supervisor_review"
    }}
  ],
  "overall_risk": "medium",
  "summary": "",
  "ask_extra_photo": false,
  "extra_photo_instruction": ""
}}
"""

    user_content = [
        {'type': 'text', 'text': prompt},
        {
            'type': 'image_url',
            'image_url': {
                'url': _image_bytes_to_data_url(image_bytes, 'image/jpeg'),
                'detail': 'high',
            },
        },
    ]

    warnings = list(fallback.get('warnings') or [])
    usage_attempts = []
    for config in configs:
        for token_limit in (1400, 1800, 2400):
            try:
                response, content, finish_reason = _run_azure_tree_completion(config, user_content, token_limit)
                token_usage = _extract_token_usage(response, config, token_limit, finish_reason, image_count=1)
                usage_attempts.append(token_usage)
                print(f"[PoleVegetation Tokens] total={token_usage['total_tokens']} prompt={token_usage['prompt_tokens']} completion={token_usage['completion_tokens']} per_image={token_usage['total_tokens_per_image']} deployment={token_usage.get('deployment')}")
                parsed = _safe_json_loads(content)
                if isinstance(parsed, dict):
                    trees = []
                    image = _decode_image(image_bytes)
                    height, width = image.shape[:2]
                    for idx, raw in enumerate(parsed.get('trees') or [], start=1):
                        trees.append(_normalize_tree_item(raw, width, height, idx))
                    wire_lines = [_normalize_line(line, width, height) for line in (parsed.get('wire_lines') or fallback.get('wire_lines') or [])]
                    wire_lines = [line for line in wire_lines if line]
                    pole_bbox = None
                    fallback_pole_bbox = fallback.get('pole_measurement', {}).get('pole_bbox') if isinstance(fallback.get('pole_measurement'), dict) else None
                    if isinstance(parsed.get('pole_measurement'), dict):
                        pole_bbox = parsed.get('pole_measurement', {}).get('pole_bbox') or fallback_pole_bbox
                    else:
                        pole_bbox = fallback_pole_bbox
                    if not pole_bbox and trees:
                        pole_bbox = fallback_pole_bbox

                    image_quality = parsed.get('image_quality') if isinstance(parsed.get('image_quality'), dict) else fallback['image_quality']
                    measurement_bundle = _build_measurement_bundle(trees, [
                        pole_bbox.get('x_min', 0.0) * width,
                        pole_bbox.get('y_min', 0.0) * height,
                        pole_bbox.get('x_max', 0.0) * width,
                        pole_bbox.get('y_max', 0.0) * height,
                    ] if pole_bbox else None, wire_lines, width, height, image_quality=image_quality)
                    result = {
                        'pole_id': '',
                        'input': fallback['input'],
                        'measurement_config': measurement_bundle['measurement_config'],
                        'image_quality': image_quality,
                        'detected_objects': parsed.get('detected_objects') if isinstance(parsed.get('detected_objects'), dict) else fallback['detected_objects'],
                        'pole_measurement': measurement_bundle['pole_measurement'],
                        'wire_lines': measurement_bundle['wire_lines'],
                        'trees': trees or fallback['trees'],
                        'tree_to_tree_distances': measurement_bundle['tree_to_tree_distances'],
                        'overall_risk': _recompute_overall_risk(trees or fallback['trees']),
                        'summary': str(parsed.get('summary') or fallback['summary']),
                        'ask_extra_photo': bool(parsed.get('ask_extra_photo', fallback['ask_extra_photo'])),
                        'extra_photo_instruction': str(parsed.get('extra_photo_instruction') or fallback['extra_photo_instruction']),
                        'safety_instructions': fallback['safety_instructions'],
                        'warnings': warnings,
                        'token_usage': dict(token_usage, status='completed'),
                    }
                    result['detected_objects']['vegetation_count'] = len(result['trees'])
                    result['summary'] = _build_summary(result['trees']) if result['trees'] else result['summary']
                    if not result['trees']:
                        return fallback
                    return result
            except Exception as exc:
                warnings.append(f'Azure context analysis unavailable: {exc}')
                break

    fallback['warnings'] = warnings or fallback['warnings']
    fallback['token_usage'] = {
        'provider': 'azure_openai',
        'status': 'failed' if usage_attempts else 'not_used',
        'image_count': 1,
        'attempts': usage_attempts,
    }
    if usage_attempts:
        last_attempt = usage_attempts[-1]
        print(f"[PoleVegetation Tokens] failed_after_attempts total={last_attempt['total_tokens']} prompt={last_attempt['prompt_tokens']} completion={last_attempt['completion_tokens']}")
    return fallback
