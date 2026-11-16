import csv
import hashlib
import importlib.util
import io
import json
import math
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from flask import abort, jsonify, render_template, request, send_file, session, url_for
from PIL import Image, ImageOps, UnidentifiedImageError

from evaluation.catalog import ROOT, MODEL_FILES, SAMPLES, LIMITATIONS, sample_by_id, homepage_evidence
from evaluation.store import JobStore
from evaluation.policy import accepted_poles, satisfies_pole_policy


def register_evaluation_routes(app):
    store = JobStore(app.config['EVALUATION_DATA_DIR'])
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='inspection')
    slots = threading.BoundedSemaphore(2)
    rate_lock, rates = threading.Lock(), {}
    app.extensions['evaluation_store'] = store

    def owner():
        session.setdefault('evaluation_owner', secrets.token_urlsafe(24))
        return session['evaluation_owner']

    def find_job(job_id):
        job = store.get(job_id, owner())
        if not job:
            abort(404)
        if job['status'] == 'done' and not satisfies_pole_policy(job['result']):
            job.update(status='failed', result=None, error='This older result does not meet the 80% pole-confidence rule. Run a new inspection.')
        return job

    def live_available():
        return app.config['LIVE_INFERENCE_ENABLED'] and all((ROOT / 'models' / m).is_file() for m in MODEL_FILES) and all(importlib.util.find_spec(m) is not None for m in ('torch', 'ultralytics', 'cv2'))

    def worker(job_id, image_bytes, source):
        try:
            store.update(job_id, status='running')
            from evaluation.inference import run_inference
            result = run_inference(image_bytes, source)
            if not satisfies_pole_policy(result):
                raise ValueError('Inference output violates the pole-confidence rule')
            result['image_url'] = f'/api/evaluation/jobs/{job_id}/image'
            store.update(job_id, status='done', result=result)
        except Exception:
            app.logger.exception('Inspection failed: %s', job_id)
            store.update(job_id, status='failed', error='The model could not complete this inspection. Try a recorded sample or retry; details are in the server log.')
        finally:
            slots.release()

    @app.get('/')
    def showcase():
        return render_template('showcase.html', evidence=homepage_evidence())

    @app.get('/demo')
    def evaluation():
        owner()
        return render_template('evaluation.html')

    @app.get('/map')
    def evaluation_map():
        owner()
        return render_template('evaluation.html', map_mode=True)

    @app.get('/api/evaluation/bootstrap')
    def evaluation_bootstrap():
        owner()
        capabilities = [
            ('image', 'Image inspection', 'demo', 'Replay real recorded predictions or run the local YOLO model stack.'),
            ('survey', 'Survey suggestions', 'demo', 'Inspect rule-based classification and generated survey answers.'),
            ('review', 'Review and export', 'demo', 'Save a review in your isolated demo and export JSON or CSV.'),
            ('map', 'Asset map', 'demo', 'Place inspected assets using operator-entered coordinates and export GeoJSON.'),
            ('video', 'Video pole selection', 'workspace', 'Track poles, select a representative frame and inspect its components in the full workspace.'),
            ('vegetation', 'Vegetation context', 'optional_provider', 'Full workspace supports optional Azure OpenAI and PlantNet integrations; credentials are required.'),
            ('training', 'Training-data export', 'workspace', 'Admin review exports corrected YOLO annotations. No model improvement is claimed here.'),
        ]
        samples = [{**{k: v for k, v in s.items() if k != 'filename'}, 'image_url': url_for('static', filename='demo_samples/' + s['filename']), 'recorded': (ROOT / 'evaluation' / 'recordings' / (s['id'] + '.json')).exists()} for s in SAMPLES]
        return jsonify(samples=samples, capabilities=[dict(zip(('id', 'title', 'status', 'description'), c)) for c in capabilities], models=[{'name': m, 'available': (ROOT / 'models' / m).is_file()} for m in MODEL_FILES], live_inference_enabled=live_available(), workspace_enabled=app.config['WORKSPACE_ENABLED'], csrf_token=session['csrf_token'], limits={'upload_mb': 10}, limitations=LIMITATIONS)

    @app.post('/api/evaluation/jobs')
    def evaluation_create_job():
        identity = owner()
        data = request.get_json(silent=True) if request.is_json else request.form
        if request.is_json and not isinstance(data, dict):
            return jsonify(error='Expected a JSON object.'), 400
        data = data or {}
        mode, sample = data.get('mode', 'recorded'), sample_by_id(data.get('sample_id'))
        if mode not in ('recorded', 'live'):
            return jsonify(error='Choose recorded or live mode.'), 400
        job_id = secrets.token_hex(16)
        if mode == 'recorded':
            if not sample:
                return jsonify(error='Choose a valid built-in sample.'), 400
            recording = ROOT / 'evaluation' / 'recordings' / (sample['id'] + '.json')
            if not recording.exists():
                return jsonify(error='A recorded run is not available for this sample. Try live inference.'), 409
            result = json.loads(recording.read_text(encoding='utf-8'))
            if result.get('provenance', {}).get('input_sha256') != hashlib.sha256((ROOT / 'static' / 'demo_samples' / sample['filename']).read_bytes()).hexdigest():
                return jsonify(error='This recording no longer matches the sample. Use live inference.'), 409
            result['provenance']['mode'] = 'recorded'
            if not satisfies_pole_policy(result):
                return jsonify(error='This recording does not meet the 80% pole-confidence rule. A new model run is required.'), 409
            result['image_url'] = url_for('static', filename='demo_samples/' + sample['filename'])
            try:
                store.add(job_id, identity, sample['id'], result=result)
            except ValueError as exc:
                return jsonify(error=str(exc)), 429
            return jsonify(id=job_id, status='done')
        if not live_available():
            return jsonify(error='Live inference is not configured on this server. Use a recorded sample.'), 503
        with rate_lock:
            now, address = time.time(), request.remote_addr or 'unknown'
            for key in list(rates):
                rates[key] = [stamp for stamp in rates[key] if stamp > now - 900]
                if not rates[key]:
                    del rates[key]
            if len(rates.get(address, [])) >= 8 or store.live_count(identity) >= 3:
                return jsonify(error='Live inspection limit reached. Retry in 15 minutes or use a recorded sample.'), 429
            if not slots.acquire(blocking=False):
                return jsonify(error='The inspection queue is busy. Retry shortly.'), 429
            rates.setdefault(address, []).append(now)
        submitted, image_path = False, None
        try:
            if sample:
                image_bytes = (ROOT / 'static' / 'demo_samples' / sample['filename']).read_bytes()
                source = sample['title']
            else:
                upload = request.files.get('image')
                if not upload:
                    return jsonify(error='Choose a sample or upload a JPG, PNG or WebP image.'), 400
                raw = upload.read(10 * 1024 * 1024 + 1)
                if len(raw) > 10 * 1024 * 1024:
                    return jsonify(error='Images must be 10 MB or smaller.'), 413
                try:
                    with Image.open(io.BytesIO(raw)) as image:
                        if image.format not in ('JPEG', 'PNG', 'WEBP') or image.width * image.height > 20_000_000:
                            return jsonify(error='Use a JPG, PNG or WebP image up to 20 megapixels.'), 400
                        normalized = ImageOps.exif_transpose(image).convert('RGB')
                        output = io.BytesIO()
                        normalized.save(output, format='JPEG', quality=93)
                        image_bytes = output.getvalue()
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                    return jsonify(error='That file is not a supported image.'), 400
                source = 'Uploaded field image'
            image_path = job_id + '.jpg'
            (store.directory / image_path).write_bytes(image_bytes)
            store.add(job_id, identity, 'live', image_path=image_path)
            executor.submit(worker, job_id, image_bytes, source)
            submitted = True
            return jsonify(id=job_id, status='queued'), 202
        except ValueError as exc:
            return jsonify(error=str(exc)), 429
        finally:
            if not submitted:
                if image_path:
                    (store.directory / image_path).unlink(missing_ok=True)
                slots.release()

    @app.get('/api/evaluation/jobs/<job_id>')
    def evaluation_job(job_id):
        job = find_job(job_id)
        return jsonify(**{k: job[k] for k in ('id', 'status', 'result', 'error', 'review')})

    @app.get('/api/evaluation/jobs/<job_id>/image')
    def evaluation_job_image(job_id):
        job = find_job(job_id)
        if not job['image_path']:
            abort(404)
        return send_file(store.directory / job['image_path'], mimetype='image/jpeg', max_age=0)

    @app.post('/api/evaluation/jobs/<job_id>/review')
    def evaluation_review(job_id):
        job = find_job(job_id)
        if job['status'] != 'done':
            return jsonify(error='Complete an inspection before reviewing it.'), 409
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error='Expected a JSON object.'), 400
        if data.get('decision') not in ('needs_review', 'approved') or not isinstance(data.get('note', ''), str) or len(data.get('note', '')) > 2000:
            return jsonify(error='Choose a valid decision and a note up to 2,000 characters.'), 400
        lat, lon = data.get('latitude'), data.get('longitude')
        if lat in ('', None) and lon in ('', None):
            lat = lon = None
        else:
            try:
                if isinstance(lat, bool) or isinstance(lon, bool):
                    raise ValueError()
                lat, lon = float(lat), float(lon)
                if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                    raise ValueError()
            except (TypeError, ValueError):
                return jsonify(error='Enter both latitude (-90 to 90) and longitude (-180 to 180), or leave both empty.'), 400
        review = {'decision': data['decision'], 'note': data.get('note', '').strip(), 'latitude': lat, 'longitude': lon, 'location_source': 'operator_entered' if lat is not None else None, 'saved_at': datetime.now(timezone.utc).isoformat()}
        store.update(job_id, review=review)
        return jsonify(status='saved', review=review)

    @app.get('/api/evaluation/assets.geojson')
    def evaluation_geojson():
        with store.connection() as conn:
            rows = conn.execute("SELECT id,source,review,result FROM jobs WHERE owner=? AND status='done' AND review IS NOT NULL AND created>=?", (owner(), time.time() - store.retention)).fetchall()
        features = []
        for row in rows:
            result = json.loads(row['result'])
            if not satisfies_pole_policy(result) or not accepted_poles(result):
                continue
            review = json.loads(row['review'])
            if review.get('latitude') is not None:
                features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [review['longitude'], review['latitude']]}, 'properties': {'id': row['id'], 'source': row['source'], 'decision': review['decision'], 'label': 'Reviewed pole inspection', 'location_source': 'operator_entered'}})
        return jsonify(type='FeatureCollection', features=features)

    @app.get('/api/evaluation/jobs/<job_id>/export')
    def evaluation_export(job_id):
        job = find_job(job_id)
        if job['status'] != 'done':
            return jsonify(error='The inspection is not finished.'), 409
        fmt = request.args.get('format', 'json')
        if fmt == 'json':
            content = json.dumps({'product': app.config['PRODUCT_NAME'], 'id': job_id, 'result': job['result'], 'review': job['review'], 'limitations': LIMITATIONS}, indent=2).encode()
            mimetype = 'application/json'
        elif fmt == 'csv':
            output = io.StringIO(newline='')
            writer = csv.writer(output)
            review = job['review'] or {}
            writer.writerow(['label', 'confidence', 'bbox', 'mode', 'decision', 'review_note', 'latitude', 'longitude'])
            def safe(value):
                text = str(value if value is not None else '')
                return "'" + text if text.startswith(('=', '+', '-', '@', '\t', '\r', '\n')) else text
            for detection in job['result'].get('detections', []):
                writer.writerow([safe(detection.get('label')), detection.get('confidence'), json.dumps(detection.get('bbox')), job['result']['provenance']['mode'], review.get('decision', ''), safe(review.get('note', '')), review.get('latitude'), review.get('longitude')])
            content, mimetype = output.getvalue().encode('utf-8-sig'), 'text/csv'
        else:
            return jsonify(error='Export format must be json or csv.'), 400
        return send_file(io.BytesIO(content), mimetype=mimetype, as_attachment=True, download_name=f'polepilot-{job_id[:8]}.{fmt}')

    @app.post('/api/evaluation/reset')
    def evaluation_reset():
        try:
            store.reset(owner())
        except ValueError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(status='reset')
