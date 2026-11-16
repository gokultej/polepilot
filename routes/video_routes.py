from flask import jsonify, request

from routes.auth_routes import login_required
from db import save_uploaded_video


def register_video_routes(app):
    @app.route('/api/predict_video', methods=['POST'])
    @login_required
    def predict_video():
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        try:
            target_fps = min(10, max(1, int(request.form.get('target_fps', 2))))
        except (ValueError, TypeError):
            return jsonify({'error': 'target_fps must be a number from 1 to 10'}), 400
        try:
            video_bytes = file.read()
            if not video_bytes:
                return jsonify({'error': 'Uploaded video is empty'}), 400
            video_id = save_uploaded_video(file.filename, file.mimetype, video_bytes)
            from video_processing import submit_task
            task_id = submit_task(video_id, target_fps=target_fps)
            return jsonify({'status': 'queued', 'task_id': task_id, 'video_id': video_id, 'message': 'Video uploaded successfully, stored in PostgreSQL, and queued for processing.'})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Failed to queue video: {exc}'}), 500

    @app.route('/api/task_status/<task_id>', methods=['GET'])
    @login_required
    def task_status(task_id):
        from video_processing import get_task_status
        status_info = get_task_status(task_id)
        if not status_info:
            return jsonify({'error': 'Task not found'}), 404
        return jsonify(status_info)
