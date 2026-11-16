"""Generate reproducible demo recordings from the actual installed model stack."""
import json
import hashlib
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.catalog import SAMPLES
from evaluation.inference import run_inference
from config import THRESHOLD_POLE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-missing', action='store_true')
    args = parser.parse_args()
    destination = ROOT / 'evaluation' / 'recordings'
    destination.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLES:
        path = ROOT / 'static' / 'demo_samples' / sample['filename']
        saved = destination / (sample['id'] + '.json')
        if args.only_missing and saved.is_file():
            old = json.loads(saved.read_text(encoding='utf-8'))
            if old.get('provenance', {}).get('input_sha256') == hashlib.sha256(path.read_bytes()).hexdigest() and old.get('provenance', {}).get('pole_confidence_threshold') == THRESHOLD_POLE:
                print(f"Keeping existing recording: {sample['id']}", flush=True)
                continue
        result = run_inference(path.read_bytes(), sample['title'])
        result['provenance']['mode'] = 'recorded'
        saved.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f"Recorded {sample['id']}: {len(result['detections'])} detections", flush=True)
