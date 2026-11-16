"""Export saved metrics from trusted local project checkpoints, not legacy JSON."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import THRESHOLD_POLE
MODELS = [
    ('pole_best6.pt', 'Pole detection'),
    ('insulator_new.pt', 'Insulator detection'),
    ('channel_12class_v2.pt', 'Hardware components'),
    ('shed_model.pt', 'Insulator shed detection'),
    ('cable_best.pt', 'Cable detection / segmentation'),
]


def main():
    models = []
    for filename, title in MODELS:
        path = ROOT / 'models' / filename
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        saved = checkpoint.get('train_metrics') or {}
        metrics = {}
        for kind, suffix in (('boxes', 'B'), ('masks', 'M')):
            values = {key: saved.get(f'metrics/{metric}({suffix})') for key, metric in [('precision', 'precision'), ('recall', 'recall'), ('map50', 'mAP50'), ('map5095', 'mAP50-95')]}
            if all(value is not None for value in values.values()):
                metrics[kind] = values
        models.append({'name': title, 'file': filename, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'checkpoint_date': checkpoint.get('date'), 'reported_split': (checkpoint.get('train_args') or {}).get('split'), **metrics})
    result = {
        'source': 'train_metrics embedded in the active local model checkpoints',
        'fresh_validation_run': False,
        'overall_application_accuracy': None,
        'validation_image_count': None,
        'independent_split_verified': False,
        'deployment_pole_min_confidence': THRESHOLD_POLE,
        'deployment_cutoff_revalidated': False,
        'note': 'Saved model-validation metrics, not an independent end-to-end application accuracy benchmark. Validation dataset size and separation from training have not been verified.',
        'training_scope': {'country': 'India', 'region': 'Andhra Pradesh', 'data': 'Real field imagery', 'source': 'Project owner description', 'outside_scope_evaluated': False},
        'models': models,
    }
    output = ROOT / 'static' / 'model_evaluation.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Exported checkpoint metrics for', len(models), 'models')


if __name__ == '__main__':
    main()
