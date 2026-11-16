from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ['pole_best6.pt', 'insulator_new.pt', 'channel_12class_v2.pt', 'shed_model.pt', 'cable_best.pt']
SAMPLES = [
    {'id': 'field-pole', 'title': 'Field pole inspection', 'description': 'Inspect a utility pole and its visible components.', 'filename': 'WhatsApp Image 2026-10-23 at 10.54.41 AM.jpeg'},
    {'id': 'line-hardware', 'title': 'Line hardware survey', 'description': 'Compare component detections with the original field photograph.', 'filename': 'WhatsApp Image 2026-05-27 at 12.30.16.jpeg'},
    {'id': 'corridor', 'title': 'Distribution corridor', 'description': 'Explore another capture angle; review uncertain and missing detections.', 'filename': 'WhatsApp Image 2026-08-26 at 4.27.47 PM.jpeg'},
    {'id': 'field-detail', 'title': 'Field hardware detail', 'description': 'A separate field photograph for inspecting visible pole hardware.', 'filename': '019c221b-81b9-7e13-b5ef-c2fe8b4f7b4d.jpg (1).jpeg'},
    {'id': 'distribution-line', 'title': 'Distribution line overview', 'description': 'Inspect the pole and line arrangement against the original photograph.', 'filename': 'PHOTO-2026-05-21-11-32-37.jpg'},
    {'id': 'mixed-hardware', 'title': 'Mixed hardware capture', 'description': 'Compare the model output across another real field capture.', 'filename': 'WhatsApp Image 2026-05-21 at 10.39.17.jpeg'},
    {'id': 'roadside-pole', 'title': 'Roadside pole survey', 'description': 'Review a roadside capture and verify the detected equipment.', 'filename': 'WhatsApp Image 2026-05-22 at 10.56.08.jpeg'},
    {'id': 'field-structure', 'title': 'Field structure inspection', 'description': 'Inspect visible structural components from a different camera position.', 'filename': 'WhatsApp Image 2026-05-26 at 10.52.31.jpeg'},
    {'id': 'line-arrangement', 'title': 'Line arrangement survey', 'description': 'Explore component detections and survey suggestions in another field image.', 'filename': 'WhatsApp Image 2026-05-27 at 11.45.19.jpeg'},
    {'id': 'alternate-angle', 'title': 'Alternate viewing angle', 'description': 'See how another viewing angle affects detected components.', 'filename': 'WhatsApp Image 2026-10-23 at 10.54.40 AM.jpeg'},
    {'id': 'field-overview', 'title': 'Field pole overview', 'description': 'Compare the full field scene with the localized model evidence.', 'filename': 'WhatsApp Image 2026-10-23 at 10.54.41 AM (1).jpeg'},
]
LIMITATIONS = [
    'Pole detections require at least 80% model confidence. Saved validation metrics have not been re-evaluated at this deployment cutoff.',
    'Detection confidence is a model score, not independently measured accuracy.',
    'Survey classification and autofill are suggestions requiring human verification.',
    'Single-image distances and voltage suggestions are not survey-grade measurements or safety clearances.',
    'Vegetation context and species identification require separately configured cloud providers in the full workspace.',
    'Training-data export is supported in the workspace; improved model performance has not been established by this demo.',
    'Map markers require operator-provided or verified GPS coordinates; photographs are not automatically geolocated.',
]


def sample_by_id(sample_id):
    return next((dict(s) for s in SAMPLES if s['id'] == sample_id), None)


def homepage_evidence():
    import json
    recordings = []
    for sample in SAMPLES:
        path = ROOT / 'evaluation' / 'recordings' / (sample['id'] + '.json')
        if path.is_file():
            recordings.append(json.loads(path.read_text(encoding='utf-8')))
    metrics_path = ROOT / 'static' / 'model_evaluation.json'
    return {
        'sample_count': len(recordings),
        'detection_count': sum(len(item.get('detections', [])) for item in recordings),
        'model_count': len(MODEL_FILES),
        'survey_fields': max((len(item.get('classification', {}).get('autofill_data', {}).get('answers', {})) for item in recordings), default=0),
        'performance': json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.is_file() else None,
    }
