"""Exercise a running local evaluator, including real live inference on request."""
import argparse
import time
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:5003')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    client = requests.Session()
    base = args.url.rstrip('/')
    bootstrap = client.get(base + '/api/evaluation/bootstrap', timeout=15)
    bootstrap.raise_for_status()
    client.headers['X-CSRF-Token'] = bootstrap.json()['csrf_token']
    for sample in bootstrap.json()['samples']:
        response = client.post(base + '/api/evaluation/jobs', json={'sample_id': sample['id'], 'mode': 'recorded'}, timeout=15)
        response.raise_for_status()
        job_id = response.json()['id']
        job = client.get(base + '/api/evaluation/jobs/' + job_id, timeout=15).json()
        assert job['status'] == 'done' and job['result']['provenance']['mode'] == 'recorded'
        print(f"Replay {sample['id']}: {len(job['result']['detections'])} detections", flush=True)
    if args.live:
        response = client.post(base + '/api/evaluation/jobs', json={'sample_id': 'field-pole', 'mode': 'live'}, timeout=15)
        response.raise_for_status()
        job_id = response.json()['id']
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            response = client.get(base + '/api/evaluation/jobs/' + job_id, timeout=15)
            response.raise_for_status()
            job = response.json()
            if job['status'] in ('done', 'failed'):
                break
            time.sleep(1)
        assert job['status'] == 'done', job
        assert job['result']['provenance']['mode'] == 'live'
        print(f"Live inference: {len(job['result']['detections'])} detections, {job['result']['provenance']['elapsed_seconds']} seconds", flush=True)
    response = client.post(base + f'/api/evaluation/jobs/{job_id}/review', json={'decision': 'needs_review', 'note': 'Automated smoke check; not a field approval.'}, timeout=15)
    response.raise_for_status()
    for fmt in ('json', 'csv'):
        response = client.get(base + f'/api/evaluation/jobs/{job_id}/export?format={fmt}', timeout=15)
        response.raise_for_status()
        assert 'attachment' in response.headers['Content-Disposition']
        print(f'Export {fmt}: OK', flush=True)
    assert client.get(base + '/api/evaluation/assets.geojson', timeout=15).json()['features'] == []
    client.post(base + '/api/evaluation/reset', json={}, timeout=15).raise_for_status()
    print('Session reset: OK', flush=True)


if __name__ == '__main__':
    main()
