from flask import jsonify, request

from routes.auth_routes import login_required
from services.pole_vegetation_report_service import (
    analyze_pole_vegetation,
    get_report,
    rerun_species_detection_for_selected_tree,
)


def register_pole_vegetation_routes(app):
    @app.route('/pole-vegetation/analyze', methods=['POST'])
    @login_required
    def pole_vegetation_analyze():
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        try:
            report = analyze_pole_vegetation(
                request.files['image'],
                latitude=request.form.get('lat') or request.form.get('latitude'),
                longitude=request.form.get('lon') or request.form.get('longitude'),
                pole_id=request.form.get('pole_id'),
                capture_date=request.form.get('capture_date'),
            )
            return jsonify(report)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(exc)}), 500

    @app.route('/pole-vegetation/select-tree', methods=['POST'])
    @login_required
    def pole_vegetation_select_tree():
        data = request.get_json(silent=True) or request.form
        report_id = data.get('report_id')
        tree_id = data.get('user_selected_tree_id') or data.get('tree_id')
        if not report_id or not tree_id:
            return jsonify({'error': 'report_id and user_selected_tree_id are required'}), 400

        try:
            report = rerun_species_detection_for_selected_tree(report_id, tree_id)
            return jsonify(report)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(exc)}), 500

    @app.route('/pole-vegetation/report/<report_id>', methods=['GET'])
    @login_required
    def pole_vegetation_report(report_id):
        report = get_report(report_id)
        if not report:
            return jsonify({'error': 'Report not found'}), 404
        return jsonify(report)
