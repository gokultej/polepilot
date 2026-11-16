import copy
import json
import tempfile
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from config import THRESHOLD_POLE, filter_pole_detections, meets_pole_confidence
from evaluation.policy import satisfies_pole_policy
from services.pole_classifier_service import classify_pole_survey


class PoleConfidenceTests(unittest.TestCase):
    def test_boundary_and_invalid_scores(self):
        self.assertEqual(THRESHOLD_POLE, .8)
        for score in (.8, .80001, 1, '0.8'):
            self.assertTrue(meets_pole_confidence(score), score)
        for score in (None, False, True, '', 'unknown', .799999, .79, 0, -1, 80, 1.01, float('nan'), float('inf')):
            self.assertFalse(meets_pole_confidence(score), score)

    def test_pole_filter_preserves_other_classes(self):
        values = [{'label': 'pole', 'confidence': .79}, {'label': 'strut_pole', 'confidence': .8}, {'label': 'ht_pin', 'confidence': .3}]
        self.assertEqual([d['label'] for d in filter_pole_detections(values)], ['strut_pole', 'ht_pin'])

    def test_low_or_missing_pole_score_cannot_trigger_survey(self):
        for label in ('pole', 'main_pole', 'strut_pole', 'ht_pole', 'lt_pole', 'support'):
            for score in (.799999, None):
                with self.subTest(label=label, score=score):
                    result = classify_pole_survey({'evaluation_mode': True, 'detections': [{'label': label, 'confidence': score}, {'label': 'ht_pin', 'confidence': .99}]})
                    self.assertFalse(result['pole_detected'])
                    self.assertIsNone(result['schema_id'])

    def test_exactly_eighty_percent_can_trigger_survey(self):
        result = classify_pole_survey({'evaluation_mode': True, 'detections': [{'label': 'pole', 'confidence': .8}, {'label': 'ht_pin', 'confidence': .9}]})
        self.assertTrue(result['pole_detected'])
        self.assertEqual(result['feature_summary']['pole_count'], 1)

    def test_pipeline_candidate_and_geometric_relabelling_obey_threshold(self):
        from pipeline import InfrastructurePipeline
        engine = InfrastructurePipeline.__new__(InfrastructurePipeline)
        detections = [{'label': 'main_pole', 'conf': score, 'box': (0, 0, 10, 100), 'polygon': None} for score in (.799999, .8)]
        with patch.object(engine, '_extract_stage_detections', return_value=detections), patch.object(engine, '_estimate_pole_angle', return_value=90):
            engine.component_model = object()
            accepted = engine._collect_pole_stage_detections([])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0][1], .8)
        poles = []
        engine._categorise('vcb', (0, 0, 10, 100), .79, 90, [], poles, [], [], [], [], defaultdict(bool))
        self.assertEqual(poles, [])

    def test_video_candidates_obey_same_boundary(self):
        import torch
        from video_processing import get_pole_detections
        boxes = [SimpleNamespace(cls=torch.tensor([0]), conf=torch.tensor([score], dtype=torch.float64), id=None, xyxy=torch.tensor([[0, 0, 10, 100]])) for score in (.79, .8)]
        result = get_pole_detections([SimpleNamespace(boxes=boxes)], {0: 'main_pole'})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['conf'], .8)

    def test_old_weak_results_cannot_be_displayed_exported_or_mapped(self):
        from app import create_app
        with tempfile.TemporaryDirectory() as directory:
            app = create_app({'TESTING': True, 'EVALUATION_DATA_DIR': directory, 'WORKSPACE_ENABLED': False, 'LIVE_INFERENCE_ENABLED': False})
            client = app.test_client()
            bootstrap = client.get('/api/evaluation/bootstrap').json
            response = client.post('/api/evaluation/jobs', json={'mode': 'recorded', 'sample_id': 'field-pole'}, headers={'X-CSRF-Token': bootstrap['csrf_token']})
            job_id = response.json['id']
            result = client.get('/api/evaluation/jobs/' + job_id).json['result']
            weak = copy.deepcopy(result)
            for detection in weak['detections']:
                if detection['label'] == 'pole':
                    detection['confidence'] = .79
            self.assertFalse(satisfies_pole_policy(weak))
            store = app.extensions['evaluation_store']
            store.update(job_id, result=weak, review={'decision': 'approved', 'latitude': 16.5, 'longitude': 80.6})
            stale = client.get('/api/evaluation/jobs/' + job_id)
            self.assertEqual(stale.json['status'], 'failed')
            self.assertIsNone(stale.json['result'])
            self.assertEqual(client.get('/api/evaluation/jobs/' + job_id + '/export').status_code, 409)
            self.assertEqual(client.get('/api/evaluation/assets.geojson').json['features'], [])
            with patch('pathlib.Path.read_text', return_value=json.dumps(weak)):
                replay = client.post('/api/evaluation/jobs', json={'mode': 'recorded', 'sample_id': 'field-pole'}, headers={'X-CSRF-Token': bootstrap['csrf_token']})
            self.assertEqual(replay.status_code, 409)


if __name__ == '__main__':
    unittest.main()
