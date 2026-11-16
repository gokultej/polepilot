import math


def normalized_bbox_to_px(bbox, image_width, image_height):
    if not isinstance(bbox, dict):
        return None
    try:
        x1 = float(bbox.get('x_min', 0.0)) * image_width
        y1 = float(bbox.get('y_min', 0.0)) * image_height
        x2 = float(bbox.get('x_max', 0.0)) * image_width
        y2 = float(bbox.get('y_max', 0.0)) * image_height
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}


def normalized_line_to_px(line, image_width, image_height):
    if not isinstance(line, dict):
        return None
    try:
        x1 = float(line.get('x1', 0.0)) * image_width
        y1 = float(line.get('y1', 0.0)) * image_height
        x2 = float(line.get('x2', 0.0)) * image_width
        y2 = float(line.get('y2', 0.0)) * image_height
    except (TypeError, ValueError):
        return None
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}


def calculate_bbox_height_px(bbox_px):
    if not bbox_px:
        return 0.0
    return max(0.0, float(bbox_px['y2']) - float(bbox_px['y1']))


def point_to_line_segment_distance_px(point, line):
    px, py = float(point[0]), float(point[1])
    x1, y1, x2, y2 = float(line['x1']), float(line['y1']), float(line['x2']), float(line['y2'])
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1), {'x': x1, 'y': y1}
    t = ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y), {'x': proj_x, 'y': proj_y}


def _point_inside_bbox(point, bbox_px):
    return bbox_px['x1'] <= point['x'] <= bbox_px['x2'] and bbox_px['y1'] <= point['y'] <= bbox_px['y2']


def _segments_intersect(a1, a2, b1, b2):
    def orientation(p, q, r):
        value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(value) < 1e-9:
            return 0
        return 1 if value > 0 else 2

    def on_segment(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(a1, b1, a2):
        return True
    if o2 == 0 and on_segment(a1, b2, a2):
        return True
    if o3 == 0 and on_segment(b1, a1, b2):
        return True
    if o4 == 0 and on_segment(b1, a2, b2):
        return True
    return False


def _line_touches_bbox(line_px, bbox_px):
    p1 = (line_px['x1'], line_px['y1'])
    p2 = (line_px['x2'], line_px['y2'])
    if _point_inside_bbox({'x': p1[0], 'y': p1[1]}, bbox_px) or _point_inside_bbox({'x': p2[0], 'y': p2[1]}, bbox_px):
        return True
    corners = [
        (bbox_px['x1'], bbox_px['y1']),
        (bbox_px['x2'], bbox_px['y1']),
        (bbox_px['x2'], bbox_px['y2']),
        (bbox_px['x1'], bbox_px['y2']),
    ]
    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]
    return any(_segments_intersect(p1, p2, edge[0], edge[1]) for edge in edges)


def bbox_to_line_distance_px(bbox_px, line_px):
    if not bbox_px or not line_px:
        return {'distance_px': None, 'is_touching': False, 'nearest_point_on_bbox': None, 'nearest_point_on_line': None}
    if _line_touches_bbox(line_px, bbox_px):
        center = {'x': (bbox_px['x1'] + bbox_px['x2']) / 2.0, 'y': (bbox_px['y1'] + bbox_px['y2']) / 2.0}
        return {'distance_px': 0.0, 'is_touching': True, 'nearest_point_on_bbox': center, 'nearest_point_on_line': center}

    candidates = [
        {'x': bbox_px['x1'], 'y': bbox_px['y1']},
        {'x': bbox_px['x2'], 'y': bbox_px['y1']},
        {'x': bbox_px['x2'], 'y': bbox_px['y2']},
        {'x': bbox_px['x1'], 'y': bbox_px['y2']},
        {'x': bbox_px['x1'], 'y': (bbox_px['y1'] + bbox_px['y2']) / 2.0},
        {'x': bbox_px['x2'], 'y': (bbox_px['y1'] + bbox_px['y2']) / 2.0},
        {'x': (bbox_px['x1'] + bbox_px['x2']) / 2.0, 'y': bbox_px['y1']},
        {'x': (bbox_px['x1'] + bbox_px['x2']) / 2.0, 'y': bbox_px['y2']},
    ]

    best_distance = None
    best_bbox_point = None
    best_line_point = None
    for point in candidates:
        distance, projected = point_to_line_segment_distance_px((point['x'], point['y']), line_px)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_bbox_point = point
            best_line_point = projected

    return {
        'distance_px': round(best_distance or 0.0, 2),
        'is_touching': False,
        'nearest_point_on_bbox': best_bbox_point,
        'nearest_point_on_line': best_line_point,
    }


def bbox_to_bbox_distance_px(bbox1_px, bbox2_px):
    if not bbox1_px or not bbox2_px:
        return None
    dx = max(bbox1_px['x1'] - bbox2_px['x2'], bbox2_px['x1'] - bbox1_px['x2'], 0.0)
    dy = max(bbox1_px['y1'] - bbox2_px['y2'], bbox2_px['y1'] - bbox1_px['y2'], 0.0)
    return round(math.hypot(dx, dy), 2)


def estimate_distance_using_pole_height(pole_height_px, distance_px, real_pole_height_m=9.0):
    if pole_height_px <= 0:
        return {
            'estimated_distance_m': None,
            'meters_per_pixel': None,
            'distance_confidence': 'low',
            'note': 'Pole height in pixels is invalid or pole is not fully visible.',
        }
    meters_per_pixel = real_pole_height_m / pole_height_px
    estimated_distance_m = distance_px * meters_per_pixel
    return {
        'estimated_distance_m': round(estimated_distance_m, 2),
        'meters_per_pixel': round(meters_per_pixel, 5),
        'distance_confidence': 'medium',
        'note': 'Estimated using assumed 9 m pole height from a single image.',
    }


def classify_clearance_risk(distance_m, is_touching=False):
    if is_touching:
        return 'critical'
    if distance_m is None:
        return 'unknown'
    if distance_m < 0.5:
        return 'high'
    if distance_m < 1.5:
        return 'medium'
    if distance_m < 3.0:
        return 'low'
    return 'no_risk'


def _clearance_label(distance_m, is_touching=False):
    if is_touching:
        return 'touching'
    if distance_m is None:
        return 'unknown'
    if distance_m < 0.5:
        return 'very_close'
    if distance_m < 1.5:
        return 'close'
    if distance_m < 3.0:
        return 'moderate'
    return 'far'


def build_distance_metrics_for_tree(tree_bbox, wire_lines, pole_bbox, image_width, image_height, real_pole_height_m=9.0):
    tree_bbox_px = normalized_bbox_to_px(tree_bbox, image_width, image_height)
    pole_bbox_px = normalized_bbox_to_px(pole_bbox, image_width, image_height) if pole_bbox else None
    wire_lines_px = [normalized_line_to_px(line, image_width, image_height) for line in (wire_lines or [])]
    wire_lines_px = [line for line in wire_lines_px if line]
    pole_height_px = calculate_bbox_height_px(pole_bbox_px) if pole_bbox_px else 0.0

    nearest_wire = {'distance_px': None, 'is_touching': False, 'nearest_point_on_bbox': None, 'nearest_point_on_line': None}
    for line_px in wire_lines_px:
        candidate = bbox_to_line_distance_px(tree_bbox_px, line_px)
        if candidate['distance_px'] is None:
            continue
        if nearest_wire['distance_px'] is None or candidate['distance_px'] < nearest_wire['distance_px']:
            nearest_wire = candidate

    wire_estimate = estimate_distance_using_pole_height(pole_height_px, nearest_wire['distance_px'] or 0.0, real_pole_height_m)
    nearest_pole_distance_px = bbox_to_bbox_distance_px(tree_bbox_px, pole_bbox_px)
    pole_estimate = estimate_distance_using_pole_height(pole_height_px, nearest_pole_distance_px or 0.0, real_pole_height_m)

    estimated_wire_clearance_m = None if nearest_wire['distance_px'] is None else wire_estimate['estimated_distance_m']
    if nearest_wire['distance_px'] is None:
        confidence = 'low'
        note = 'Distance could not be estimated because pole or wire geometry is incomplete. Risk is based on visual proximity only.'
    else:
        confidence = wire_estimate['distance_confidence']
        note = wire_estimate['note']
        if not pole_bbox_px or pole_height_px < image_height * 0.25:
            confidence = 'low'
            note = 'Distance could not be estimated reliably because pole geometry is incomplete. Risk is based on visual proximity only.'
            estimated_wire_clearance_m = None
    if nearest_wire['is_touching']:
        estimated_wire_clearance_m = 0.0
        confidence = 'medium'
        note = 'Estimated using assumed 9 m pole height from a single image. Not survey-grade.'

    return {
        'pole_height_assumed_m': real_pole_height_m,
        'pole_height_px': round(pole_height_px, 2) if pole_height_px else 0.0,
        'meters_per_pixel': wire_estimate['meters_per_pixel'] if estimated_wire_clearance_m is not None else None,
        'nearest_wire_distance_px': nearest_wire['distance_px'],
        'estimated_wire_clearance_m': estimated_wire_clearance_m,
        'nearest_pole_distance_px': nearest_pole_distance_px,
        'estimated_pole_distance_m': pole_estimate['estimated_distance_m'] if nearest_pole_distance_px is not None and estimated_wire_clearance_m is not None else None,
        'clearance_label': _clearance_label(estimated_wire_clearance_m, nearest_wire['is_touching']),
        'distance_confidence': confidence,
        'distance_note': note if note.endswith('Not survey-grade.') else note + ' Not survey-grade.',
        'is_touching_wire': nearest_wire['is_touching'],
        'nearest_tree_point_px': nearest_wire['nearest_point_on_bbox'],
        'nearest_wire_point_px': nearest_wire['nearest_point_on_line'],
        'recomputed_risk_level': classify_clearance_risk(estimated_wire_clearance_m, nearest_wire['is_touching']),
    }
