"""Check the integrity behind publicly displayed sample and model statistics."""
import hashlib
import json
import unittest

from evaluation.catalog import ROOT, SAMPLES, homepage_evidence


class DemoEvidenceTests(unittest.TestCase):
    def test_all_catalog_images_are_unique_and_match_recorded_inputs(self):
        seen = set()
        total = 0
        for sample in SAMPLES:
            digest = hashlib.sha256((ROOT / 'static' / 'demo_samples' / sample['filename']).read_bytes()).hexdigest()
            self.assertNotIn(digest, seen, 'Duplicate photographs must not inflate the sample count')
            seen.add(digest)
            result = json.loads((ROOT / 'evaluation' / 'recordings' / (sample['id'] + '.json')).read_text())
            self.assertEqual(result['provenance']['input_sha256'], digest)
            self.assertEqual(result['provenance']['mode'], 'recorded')
            self.assertEqual(result['provenance']['pole_confidence_threshold'], .8)
            from evaluation.policy import satisfies_pole_policy
            self.assertTrue(satisfies_pole_policy(result))
            total += len(result['detections'])
        self.assertEqual(homepage_evidence()['sample_count'], len(seen))
        self.assertEqual(homepage_evidence()['detection_count'], total)

    def test_metric_artifact_matches_actual_model_files_and_replays(self):
        artifact = json.loads((ROOT / 'static' / 'model_evaluation.json').read_text())
        self.assertIsNone(artifact['overall_application_accuracy'])
        self.assertFalse(artifact['independent_split_verified'])
        self.assertFalse(artifact['fresh_validation_run'])
        hashes = {m['file']: m['sha256'] for m in artifact['models']}
        for filename, digest in hashes.items():
            self.assertEqual(hashlib.sha256((ROOT / 'models' / filename).read_bytes()).hexdigest(), digest)
        for sample in SAMPLES:
            replay = json.loads((ROOT / 'evaluation' / 'recordings' / (sample['id'] + '.json')).read_text())
            self.assertEqual(replay['provenance']['model_sha256'], hashes)
        for model in artifact['models']:
            for kind in ('boxes', 'masks'):
                for value in model.get(kind, {}).values():
                    self.assertTrue(0 <= value <= 1)

    def test_every_recording_can_be_replayed_through_the_isolated_api(self):
        import tempfile
        from app import create_app
        with tempfile.TemporaryDirectory() as directory:
            app = create_app({'TESTING': True, 'EVALUATION_DATA_DIR': directory, 'WORKSPACE_ENABLED': False, 'LIVE_INFERENCE_ENABLED': False})
            client = app.test_client()
            token = client.get('/api/evaluation/bootstrap').json['csrf_token']
            for sample in SAMPLES:
                response = client.post('/api/evaluation/jobs', json={'sample_id': sample['id'], 'mode': 'recorded'}, headers={'X-CSRF-Token': token})
                self.assertEqual(response.status_code, 200, sample['id'])
                job = client.get('/api/evaluation/jobs/' + response.json['id']).json
                self.assertEqual(job['status'], 'done')


if __name__ == '__main__':
    unittest.main()
