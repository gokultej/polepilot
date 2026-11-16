"""
routes/survey_routes.py
Flask route handler for Pole Survey Classification & Auto-Fill Redirection.
"""

from flask import jsonify, request
import os
import json

from config import ENABLE_POLE_AUTO_SURVEY_REDIRECT, BASE_DIR
from services.pole_classifier_service import classify_pole_survey
from routes.auth_routes import login_required, admin_required


def register_survey_routes(app):
    @app.route('/api/survey/config', methods=['GET'])
    def get_survey_config():
        return jsonify({
            "auto_survey_redirect_enabled": ENABLE_POLE_AUTO_SURVEY_REDIRECT
        })

    @app.route('/api/survey/classify', methods=['POST'])
    def classify_survey():
        if not ENABLE_POLE_AUTO_SURVEY_REDIRECT:
            return jsonify({
                "status": "DISABLED",
                "message": "Auto-survey redirection is currently disabled.",
                "auto_survey_redirect_enabled": False
            }), 200

        data = request.json or {}
        classification = classify_pole_survey(data)

        # Load schema if pole detected
        schema_data = None
        if classification.get("pole_detected") and classification.get("schema_id"):
            schema_file_path = os.path.join(BASE_DIR, "11kv_ht_pole_survey_schema.json")
            if os.path.exists(schema_file_path):
                try:
                    with open(schema_file_path, "r", encoding="utf-8") as f:
                        schema_data = json.load(f)
                except Exception as e:
                    print("Failed to load schema file:", e)

        return jsonify({
            "auto_survey_redirect_enabled": True,
            "classification": classification,
            "schema": schema_data
        })

    @app.route('/api/survey/submit', methods=['POST'])
    @login_required
    def submit_survey():
        data = request.json or {}
        from services.pole_autofill_service import store_submitted_survey
        result = store_submitted_survey(data)
        return jsonify(result), 200

    @app.route('/api/survey/list', methods=['GET'])
    @admin_required
    def list_surveys():
        from services.pole_autofill_service import get_stored_surveys
        surveys = get_stored_surveys()
        return jsonify({"status": "SUCCESS", "surveys": surveys}), 200
