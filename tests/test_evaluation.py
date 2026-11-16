"""Exercise evaluator boundaries without loading models or touching company data."""
import io
import json
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
from app import create_app


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app({'TESTING': True, 'EVALUATION_DATA_DIR': self.tmp.name, 'LIVE_INFERENCE_ENABLED': False, 'WORKSPACE_ENABLED': False})
        self.client = self.app.test_client()
        self.token = self.client.get('/api/evaluation/bootstrap').json['csrf_token']

    def tearDown(self):
        self.tmp.cleanup()

    def post(self, path, data, client=None, token=None):
        return (client or self.client).post(path, json=data, headers={'X-CSRF-Token': token or self.token})

    def replay(self):
        response = self.post('/api/evaluation/jobs', {'mode': 'recorded', 'sample_id': 'field-pole'})
        self.assertEqual(response.status_code, 200)
        return response.json['id']

    def test_public_pages_and_health(self):
        for path in ('/', '/demo', '/map', '/health/live', '/health/ready'):
            self.assertEqual(self.client.get(path).status_code, 200)
        self.assertEqual(self.client.get('/workspace').location, '/demo')
        self.assertIn(b'PolePilot', self.client.get('/').data)

    def test_csrf_required(self):
        self.assertEqual(self.client.post('/api/evaluation/jobs', json={}).status_code, 403)
        self.assertEqual(self.client.post('/api/evaluation/reset').status_code, 403)

    def test_replay_provenance(self):
        job = self.client.get('/api/evaluation/jobs/' + self.replay()).json
        self.assertEqual(job['status'], 'done')
        self.assertEqual(job['result']['provenance']['mode'], 'recorded')
        self.assertEqual(len(job['result']['provenance']['model_sha256']), 5)
        self.assertGreater(len(job['result']['detections']), 0)

    def test_demo_autofill_does_not_read_company_sequence(self):
        from services.pole_autofill_service import populate_survey_answers
        with patch('services.pole_autofill_service.get_next_pole_location_number', side_effect=AssertionError('Company DB must not be read')):
            result = populate_survey_answers({'evaluation_mode': True, 'detections': []})
        self.assertEqual(result['answers']['pole_location_number'], '')
        self.assertEqual(result['answers']['pole_to_pole_distance'], '')

    def test_session_isolation_all_endpoints(self):
        job_id = self.replay()
        other = self.app.test_client()
        token = other.get('/api/evaluation/bootstrap').json['csrf_token']
        for suffix in ('', '/image', '/export?format=json'):
            self.assertEqual(other.get('/api/evaluation/jobs/' + job_id + suffix).status_code, 404)
        self.assertEqual(self.post('/api/evaluation/jobs/' + job_id + '/review', {'decision': 'approved'}, other, token).status_code, 404)

    def test_review_coordinates_and_geojson(self):
        job_id = self.replay()
        review = self.post('/api/evaluation/jobs/' + job_id + '/review', {'decision': 'needs_review', 'note': 'Verify on site', 'latitude': 17.4, 'longitude': 78.5})
        self.assertEqual(review.status_code, 200)
        features = self.client.get('/api/evaluation/assets.geojson').json['features']
        self.assertEqual(features[0]['geometry']['coordinates'], [78.5, 17.4])
        self.assertEqual(features[0]['properties']['location_source'], 'operator_entered')
        self.assertEqual(self.app.test_client().get('/api/evaluation/assets.geojson').json['features'], [])

    def test_unlocated_results_do_not_create_markers(self):
        job_id = self.replay()
        self.post('/api/evaluation/jobs/' + job_id + '/review', {'decision': 'approved'})
        self.assertEqual(self.client.get('/api/evaluation/assets.geojson').json['features'], [])

    def test_invalid_review_and_coordinate_ranges(self):
        path = '/api/evaluation/jobs/' + self.replay() + '/review'
        for data in ({'decision': 'deleted'}, {'decision': 'approved', 'note': 5}, {'decision': 'approved', 'note': 'x' * 2001}, {'decision': 'approved', 'latitude': 91, 'longitude': 4}, {'decision': 'approved', 'latitude': 1}, {'decision': 'approved', 'latitude': True, 'longitude': 2}, {'decision': 'approved', 'latitude': 'nan', 'longitude': 2}):
            with self.subTest(data=str(data)[:80]):
                self.assertEqual(self.post(path, data).status_code, 400)

    def test_export_json_and_csv_injection_protection(self):
        job_id = self.replay()
        self.post('/api/evaluation/jobs/' + job_id + '/review', {'decision': 'approved', 'note': '=HYPERLINK("bad")'})
        response = self.client.get(f'/api/evaluation/jobs/{job_id}/export?format=json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data)['product'], 'PolePilot')
        self.assertIn('attachment', response.headers['Content-Disposition'])
        csv_response = self.client.get(f'/api/evaluation/jobs/{job_id}/export?format=csv')
        self.assertIn("'=HYPERLINK", csv_response.data.decode('utf-8-sig'))
        self.assertEqual(self.client.get(f'/api/evaluation/jobs/{job_id}/export?format=pdf').status_code, 400)

    def test_reset_deletes_only_own_jobs(self):
        job_id = self.replay()
        self.assertEqual(self.post('/api/evaluation/reset', {}).status_code, 200)
        self.assertEqual(self.client.get('/api/evaluation/jobs/' + job_id).status_code, 404)

    def test_invalid_job_input(self):
        for data in ([], {'mode': 'fake'}, {'sample_id': '../../config.py'}, {'sample_id': 'missing', 'mode': 'recorded'}):
            self.assertEqual(self.post('/api/evaluation/jobs', data).status_code, 400)
        self.assertEqual(self.post('/api/evaluation/jobs', {'mode': 'live', 'sample_id': 'field-pole'}).status_code, 503)

    def test_live_worker_with_stubbed_inference(self):
        self.app.config['LIVE_INFERENCE_ENABLED'] = True
        result = {'detections': [], 'width': 2, 'height': 2, 'master': {}, 'provenance': {'mode': 'live'}, 'classification': {}}
        with patch('evaluation.inference.run_inference', return_value=result):
            response = self.post('/api/evaluation/jobs', {'mode': 'live', 'sample_id': 'field-pole'})
            self.assertEqual(response.status_code, 202)
            job_id = response.json['id']
            for _ in range(100):
                job = self.client.get('/api/evaluation/jobs/' + job_id).json
                if job['status'] in ('done', 'failed'):
                    break
                time.sleep(.01)
            self.assertEqual(job['status'], 'done')
            self.assertEqual(self.client.get(job['result']['image_url']).status_code, 200)

    def test_upload_rejects_disguised_file(self):
        self.app.config['LIVE_INFERENCE_ENABLED'] = True
        response = self.client.post('/api/evaluation/jobs', data={'mode': 'live', 'image': (io.BytesIO(b'not-an-image'), 'fake.jpg')}, headers={'X-CSRF-Token': self.token})
        self.assertEqual(response.status_code, 400)

    def test_full_workspace_registration_and_auth_boundaries(self):
        # Register real legacy routes, but never initialize or write the company DB.
        with patch('db.init_postgres_db'):
            workspace_app = create_app({'TESTING': True, 'EVALUATION_DATA_DIR': self.tmp.name + '/workspace-test', 'WORKSPACE_ENABLED': True})
        client = workspace_app.test_client()
        response = client.get('/workspace')
        self.assertTrue(response.location.endswith('/login'))
        self.assertEqual(client.get('/demo_login').location, '/demo')
        login = client.get('/login')
        self.assertIn(b'name="csrf_token"', login.data)
        with client.session_transaction() as state:
            state['user'], state['role'] = 'test-worker', 'user'
            token = state['csrf_token']
        self.assertEqual(client.get('/workspace').status_code, 200)
        response = client.post('/api/update_asset_status', json={'asset_id': 'any', 'status': 'approved'}, headers={'X-CSRF-Token': token})
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.location.endswith('/workspace'))
        self.assertEqual(client.get('/api/survey/list').status_code, 302)

    def test_upload_image_sanitized_and_private(self):
        self.app.config['LIVE_INFERENCE_ENABLED'] = True
        buffer = io.BytesIO()
        Image.new('RGB', (3, 3)).save(buffer, format='PNG')
        buffer.seek(0)
        result = {'detections': [], 'width': 3, 'height': 3, 'provenance': {'mode': 'live'}}
        with patch('evaluation.inference.run_inference', return_value=result):
            response = self.client.post('/api/evaluation/jobs', data={'mode': 'live', 'image': (buffer, '../../unsafe.png')}, headers={'X-CSRF-Token': self.token})
            self.assertEqual(response.status_code, 202)
            job_id = response.json['id']
            for _ in range(100):
                job = self.client.get('/api/evaluation/jobs/' + job_id).json
                if job['status'] in ('done', 'failed'):
                    break
                time.sleep(.01)
            image_response = self.client.get('/api/evaluation/jobs/' + job_id + '/image')
            self.assertEqual(image_response.mimetype, 'image/jpeg')
            self.assertEqual(self.app.test_client().get('/api/evaluation/jobs/' + job_id + '/image').status_code, 404)


if __name__ == '__main__':
    unittest.main()
