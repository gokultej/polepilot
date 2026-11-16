import base64
import io

from flask import jsonify, request, send_file

from routes.auth_routes import login_required
from db import clean_b64, get_db_connection, parse_db_json, save_uploaded_image, hydrate_asset_image_record
from image_processing import process_image_file


def register_image_routes(app):
    @app.route('/predict_stream', methods=['POST'])
    @login_required
    def predict_stream():
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image payload'}), 400
        img_b64 = data['image'].split(',').pop() if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(img_b64)
        uploaded_image_id = save_uploaded_image('predict_stream.jpg', 'image/jpeg', image_bytes)
        file_stream = io.BytesIO(image_bytes)
        try:
            result = process_image_file(file_stream)
            result.pop('annotated_image', None)
            result['uploaded_image_id'] = uploaded_image_id
            return jsonify(result)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Inference Error: {exc}'}), 500

    @app.route('/predict', methods=['POST'])
    @login_required
    def predict():
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        try:
            uploaded = request.files['image']
            image_bytes = uploaded.read()
            uploaded_image_id = save_uploaded_image(uploaded.filename or 'predict.jpg', uploaded.mimetype or 'image/jpeg', image_bytes)
            result = process_image_file(io.BytesIO(image_bytes))
            result['uploaded_image_id'] = uploaded_image_id
            return jsonify(result)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Inference Error: {exc}'}), 500

    @app.route('/api/download_annotated/<asset_id>')
    @login_required
    def download_annotated(asset_id):
        conn = get_db_connection()
        from flask import session
        row = conn.execute('SELECT i.* FROM asset_images i JOIN assets a ON a.id=i.asset_id WHERE i.asset_id=? AND (a.worker_name=? OR ?=?) LIMIT 1', (asset_id, session['user'], session.get('role'), 'admin')).fetchone()
        conn.close()
        if not row:
            return 'Asset image not found', 404
        img_dict = hydrate_asset_image_record(row)
        from routes.admin_routes import annotate_image
        annotated_b64 = annotate_image(img_dict.get('image_b64', ''), img_dict.get('detections', []))
        try:
            img_data = base64.b64decode(clean_b64(annotated_b64))
            return send_file(io.BytesIO(img_data), mimetype='image/jpeg', as_attachment=True, download_name=f'Annotated_Asset_{asset_id[:8]}.jpg')
        except Exception as exc:
            print('Base64 decode failed:', exc)
            return jsonify({'error': 'Invalid image data'}), 400

    @app.route('/api/demo_samples', methods=['GET'])
    def demo_samples():
        """Return list of built-in demo sample images and videos from static/demo_samples/"""
        import os
        samples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'demo_samples')
        if not os.path.exists(samples_dir):
            return jsonify({'samples': [], 'videos': [], 'message': 'No demo samples directory found'})

        images = []
        videos = []
        img_idx = 1
        vid_idx = 1
        for fname in sorted(os.listdir(samples_dir)):
            low = fname.lower()
            if low.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                images.append({
                    'filename': fname,
                    'url': f'/static/demo_samples/{fname}',
                    'label': f'Pole Sample {img_idx}',
                    'type': 'image'
                })
                img_idx += 1
            elif low.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append({
                    'filename': fname,
                    'url': f'/static/demo_samples/{fname}',
                    'label': f'Drone/Survey Video {vid_idx}',
                    'type': 'video'
                })
                vid_idx += 1

        return jsonify({'samples': images, 'videos': videos})
