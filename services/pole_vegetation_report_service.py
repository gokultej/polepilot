import json
import uuid
from datetime import datetime
from pathlib import Path

from db import get_db_connection, save_uploaded_image
from services.image_crop_service import crop_image_by_normalized_bbox
from services.plantnet_service import identify_species
from services.vegetation_context_service import analyze_pole_context_with_azure, infer_season


UPLOAD_ROOT = Path('uploads') / 'pole_vegetation'
RISKY_LEVELS = {'medium', 'high', 'critical'}


def _now_iso():
    return datetime.now().isoformat()


def _ensure_plantnet_shape(tree):
    tree.setdefault('plantnet_result', {
        'best_match': '',
        'scientific_name': '',
        'common_name': '',
        'confidence': 0.0,
        'top_predictions': [],
    })


def _report_table_upsert(report_id, pole_id, latitude, longitude, capture_date, season, original_image_path, original_uploaded_image_id, azure_context_json, report_json, user_selected_tree_id=None):
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO pole_vegetation_reports (
                id, pole_id, latitude, longitude, capture_date, season, created_at,
                original_image_path, original_uploaded_image_id, azure_context_json, report_json, user_selected_tree_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                pole_id = excluded.pole_id,
                latitude = excluded.latitude,
                longitude = excluded.longitude,
                capture_date = excluded.capture_date,
                season = excluded.season,
                original_image_path = excluded.original_image_path,
                original_uploaded_image_id = excluded.original_uploaded_image_id,
                azure_context_json = excluded.azure_context_json,
                report_json = excluded.report_json,
                user_selected_tree_id = excluded.user_selected_tree_id
            """,
            (
                report_id, pole_id, latitude, longitude, capture_date, season, _now_iso(),
                original_image_path, original_uploaded_image_id, azure_context_json, report_json, user_selected_tree_id
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_report(report_id):
    conn = get_db_connection()
    try:
        row = conn.execute('SELECT * FROM pole_vegetation_reports WHERE id = ?', (report_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    data = json.loads(row['report_json']) if row.get('report_json') else {}
    data['report_id'] = row['id']
    return data


def _save_report(report_id, report):
    _report_table_upsert(
        report_id=report_id,
        pole_id=report.get('pole_id') or '',
        latitude=report.get('input', {}).get('latitude'),
        longitude=report.get('input', {}).get('longitude'),
        capture_date=report.get('input', {}).get('capture_date'),
        season=report.get('input', {}).get('season'),
        original_image_path=report.get('original_image_path') or '',
        original_uploaded_image_id=report.get('original_uploaded_image_id') or '',
        azure_context_json=json.dumps(report.get('azure_context') or {}, ensure_ascii=True),
        report_json=json.dumps(report, ensure_ascii=True),
        user_selected_tree_id=report.get('user_selected_tree_id') or '',
    )


def _apply_species_detection(tree, original_image_path, report_dir):
    _ensure_plantnet_shape(tree)
    crop_path = tree.get('crop_image_path')
    if not crop_path:
        crop_path = str(Path(report_dir) / f"{tree['tree_id']}.jpg")
        crop_path = crop_image_by_normalized_bbox(original_image_path, tree['bbox_normalized'], crop_path)
        tree['crop_image_path'] = crop_path
    plantnet_result = identify_species(crop_path)
    tree['plantnet_result'] = plantnet_result
    confidence = float(plantnet_result.get('confidence') or 0.0)
    if confidence < 0.35:
        tree['recommended_action'] = 'capture_side_angle' if tree['risk_level'] in RISKY_LEVELS else tree.get('recommended_action', 'monitor')
    return tree


def analyze_pole_vegetation(image_storage, latitude=None, longitude=None, pole_id=None, capture_date=None):
    report_id = str(uuid.uuid4())
    report_dir = UPLOAD_ROOT / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    image_bytes = image_storage.read()
    if not image_bytes:
        raise ValueError('Uploaded image is empty')

    capture_date = capture_date or _now_iso()
    season = infer_season(capture_date, latitude)
    original_name = image_storage.filename or 'pole_context.jpg'
    original_path = report_dir / 'original.jpg'
    original_path.write_bytes(image_bytes)
    uploaded_image_id = save_uploaded_image(original_name, image_storage.mimetype or 'image/jpeg', image_bytes, source_path=str(original_path))

    context = analyze_pole_context_with_azure(image_bytes, lat=latitude, lon=longitude, date=capture_date, season=season)
    trees = context.get('trees') or []

    for tree in trees:
        _ensure_plantnet_shape(tree)
        if tree.get('species_detection_required'):
            try:
                _apply_species_detection(tree, original_path, report_dir)
            except Exception as exc:
                warnings = context.setdefault('warnings', [])
                warnings.append(f"Species detection skipped for {tree.get('tree_id')}: {exc}")

    if not context.get('ask_extra_photo'):
        low_conf_risky_tree = next(
            (
                tree for tree in trees
                if tree.get('risk_level') in RISKY_LEVELS
                and (tree.get('distance_metrics') or {}).get('estimated_wire_clearance_m') is None
            ),
            None,
        )
        if low_conf_risky_tree:
            context['ask_extra_photo'] = True
            context['extra_photo_instruction'] = 'Capture one side-angle photo from a safe distance because pole or wire geometry is incomplete for distance estimation.'

    report = {
        'report_id': report_id,
        'pole_id': pole_id or '',
        'input': {
            'latitude': float(latitude) if latitude not in (None, '') else 0.0,
            'longitude': float(longitude) if longitude not in (None, '') else 0.0,
            'capture_date': capture_date,
            'season': season,
        },
        'measurement_config': context.get('measurement_config') or {},
        'image_quality': context.get('image_quality') or {},
        'detected_objects': context.get('detected_objects') or {},
        'pole_measurement': context.get('pole_measurement') or {},
        'wire_lines': context.get('wire_lines') or [],
        'trees': trees,
        'tree_to_tree_distances': context.get('tree_to_tree_distances') or [],
        'overall_risk': context.get('overall_risk') or 'no_risk',
        'summary': context.get('summary') or '',
        'ask_extra_photo': bool(context.get('ask_extra_photo')),
        'extra_photo_instruction': context.get('extra_photo_instruction') or '',
        'safety_instructions': context.get('safety_instructions') or [],
        'warnings': context.get('warnings') or [],
        'token_usage': context.get('token_usage') or {},
        'original_image_path': str(original_path),
        'original_uploaded_image_id': uploaded_image_id,
        'azure_context': context,
        'user_selected_tree_id': '',
    }

    _save_report(report_id, report)
    return report


def rerun_species_detection_for_selected_tree(report_id, user_selected_tree_id):
    report = get_report(report_id)
    if not report:
        raise ValueError('Report not found')

    trees = report.get('trees') or []
    target = next((tree for tree in trees if tree.get('tree_id') == user_selected_tree_id), None)
    if not target:
        raise ValueError('Selected tree was not found in this report')

    original_path = report.get('original_image_path')
    if not original_path or not Path(original_path).exists():
        raise ValueError('Original image for this report is not available')

    report_dir = Path(original_path).parent
    target['species_detection_required'] = True
    _apply_species_detection(target, original_path, report_dir)
    report['user_selected_tree_id'] = user_selected_tree_id

    plantnet_conf = float(target.get('plantnet_result', {}).get('confidence') or 0.0)
    if plantnet_conf < 0.35:
        report['ask_extra_photo'] = True
        report['extra_photo_instruction'] = 'Capture one leaf close-up of the selected tree from a safe distance if possible.'

    _save_report(report_id, report)
    return report
