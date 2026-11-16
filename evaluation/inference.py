"""Keep model imports off the public web startup path; serialize model access."""
import hashlib
import io
import threading
import time
from datetime import datetime, timezone

from evaluation.catalog import ROOT, MODEL_FILES
from config import THRESHOLD_POLE

INFERENCE_LOCK = threading.Lock()


def run_inference(image_bytes, source):
    with INFERENCE_LOCK:
        started = time.perf_counter()
        from image_processing import process_image_file
        from services.pole_classifier_service import classify_pole_survey

        result = process_image_file(io.BytesIO(image_bytes))
        result.pop('annotated_image', None)
        # Existing classifier expects 'pole', while raw models return main_pole.
        payload = {'evaluation_mode': True, 'detections': [dict(d, label='pole') if d.get('label', '').lower() in ('main_pole', 'strut_pole') else d for d in result['detections']]}
        result['classification'] = classify_pole_survey(payload)
        result['provenance'] = {
            'mode': 'live', 'source': source,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': round(time.perf_counter() - started, 2),
            'input_sha256': hashlib.sha256(image_bytes).hexdigest(),
            'model_files': MODEL_FILES,
            'pole_confidence_threshold': THRESHOLD_POLE,
            'model_sha256': {name: hashlib.sha256((ROOT / 'models' / name).read_bytes()).hexdigest() for name in MODEL_FILES},
        }
        return result
