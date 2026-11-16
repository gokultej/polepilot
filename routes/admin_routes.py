from datetime import datetime
import json
from io import BytesIO
import base64
import re
import urllib.parse

import cv2
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

from flask import jsonify, render_template, request, send_file, session

from routes.auth_routes import admin_required, login_required
from db import get_db_connection, hydrate_asset_image_record, log_activity, parse_db_json
from training_pipeline import build_training_dashboard_stats, export_asset_to_training, get_training_stats


def clean_report_b64(b64_str):
    if not b64_str:
        return ""
    b64_str = str(b64_str).strip()
    b64_str = urllib.parse.unquote(b64_str)
    if 'base64,' in b64_str:
        b64_str = b64_str.split('base64,')[-1]
    elif ',' in b64_str:
        b64_str = b64_str.split(',')[-1]
    b64_str = "".join(b64_str.split()).replace('-', '+').replace('_', '/')
    b64_str = re.sub(r'[^a-zA-Z0-9+/=]', '', b64_str)
    b64_str = b64_str.rstrip('=')
    missing_padding = len(b64_str) % 4
    if missing_padding:
        b64_str += '=' * (4 - missing_padding)
    return b64_str


def get_image_bytes(image_data):
    if not image_data:
        return None
    try:
        return base64.b64decode(clean_report_b64(image_data))
    except Exception:
        return None


def annotate_image(image_b64, detections):
    try:
        img_data = get_image_bytes(image_b64)
        if not img_data:
            return image_b64
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_b64
        for det in detections:
            label = det.get('label', 'object').upper()
            box = det.get('bbox')
            poly = det.get('polygon')
            color = (255, 0, 0)
            if 'INSULATOR' in label:
                color = (255, 0, 255)
            if 'POLE' in label:
                color = (139, 0, 139)
            if 'CROSSARM' in label:
                color = (0, 255, 0)
            if 'CONDUCTOR' in label:
                color = (0, 165, 255)
            if poly:
                pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, color, 2)
            if box:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label_txt = f"{label} {int(det.get('confidence', 1.0) * 100)}%"
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                cv2.putText(img, label_txt, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as exc:
        print(f"Annotation failed: {exc}")
        return image_b64


def generate_asset_pdf(asset_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor("#1e293b"), spaceAfter=10, fontName='Helvetica-Bold')
    label_style = ParagraphStyle('LabelStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#64748b"), textTransform='uppercase', fontName='Helvetica-Bold')
    story = []
    story.append(Paragraph("ASAKTA VISION AI", styles['Heading4']))
    story.append(Paragraph("INFRASTRUCTURE INSPECTION REPORT", title_style))
    story.append(Paragraph(f"Asset ID: {asset_data['id']}", styles['Normal']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    summary_data = [
        [Paragraph("<b>WORKER</b>", label_style), asset_data['worker_name']],
        [Paragraph("<b>CLASSIFICATION</b>", label_style), asset_data['asset_class']],
        [Paragraph("<b>VOLTAGE</b>", label_style), asset_data['voltage']],
        [Paragraph("<b>TIMESTAMP</b>", label_style), asset_data['timestamp']],
        [Paragraph("<b>STATUS</b>", label_style), asset_data['status']],
    ]
    table = Table(summary_data, colWidths=[120, 350])
    table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke), ('PADDING', (0, 0), (-1, -1), 10)]))
    story.append(table)
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>DIAGNOSTIC SUMMARY</b>", label_style))
    story.append(Paragraph(asset_data['reason'], styles['Italic']))
    story.append(Spacer(1, 30))
    story.append(Paragraph("IMAGE EVIDENCE GALLERY", styles['Heading2']))
    story.append(Spacer(1, 10))
    for i, img in enumerate(asset_data['images']):
        story.append(Paragraph(f"Frame #{i + 1} Analysis", styles['Heading3']))
        annotated_b64 = annotate_image(img['image_b64'], img['detections'])
        img_data = get_image_bytes(annotated_b64)
        if img_data:
            try:
                story.append(Image(BytesIO(img_data), width=450, height=300, kind='proportional'))
            except Exception as exc:
                print(f"Skipping corrupt image in PDF: {exc}")
                story.append(Paragraph("[ Image Unavailable / Corrupted ]", styles['Italic']))
        else:
            story.append(Paragraph("[ Image Data Missing ]", styles['Italic']))
        story.append(Spacer(1, 15))
        story.append(Paragraph("Components Identified:", label_style))
        det_rows = [["Label", "Confidence", "Type"]]
        for det in img['detections']:
            det_rows.append([det['label'], f"{det.get('confidence', 1.0) * 100:.1f}%", "Manual" if det.get('manual') else "AI"])
        det_table = Table(det_rows, colWidths=[150, 100, 100])
        det_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3b82f6")), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('FONTSIZE', (0, 0), (-1, -1), 8)]))
        story.append(det_table)
        story.append(Spacer(1, 30))
        if (i + 1) % 2 == 0:
            from reportlab.platypus import PageBreak
            story.append(PageBreak())
    doc.build(story)
    buffer.seek(0)
    return buffer


def generate_asset_excel(asset_data):
    df_data = []
    for i, img in enumerate(asset_data['images']):
        for det in img['detections']:
            df_data.append({
                "Asset_ID": asset_data['id'],
                "Worker": asset_data['worker_name'],
                "Timestamp": asset_data['timestamp'],
                "Frame_No": i + 1,
                "Component": det['label'],
                "Confidence": det.get('confidence', 1.0),
                "Category": asset_data['asset_class'],
                "Voltage": asset_data['voltage'],
            })
    df = pd.DataFrame(df_data)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Detection_Log')
    buffer.seek(0)
    return buffer


def generate_global_pdf(assets_list):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor("#1e293b"), spaceAfter=10, fontName='Helvetica-Bold')
    story = []
    story.append(Paragraph("ASAKTA VISION AI", styles['Heading4']))
    story.append(Paragraph("GLOBAL INFRASTRUCTURE REPORT", title_style))
    story.append(Paragraph(f"Total Assets: {len(assets_list)}", styles['Normal']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    if not assets_list:
        story.append(Paragraph("No assets found in the database.", styles['Normal']))
        doc.build(story)
        buffer.seek(0)
        return buffer
    data = [["Asset ID", "Worker", "Status", "Class", "Timestamp"]]
    for asset in assets_list:
        short_id = asset['id'][:8] + "..." if len(asset['id']) > 8 else asset['id']
        data.append([short_id, asset['worker_name'], asset['status'], asset['asset_class'] or "N/A", asset['timestamp']])
    table = Table(data, colWidths=[100, 100, 60, 100, 140])
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3b82f6")), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('FONTSIZE', (0, 0), (-1, 0), 10), ('FONTSIZE', (0, 1), (-1, -1), 8)]))
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer


def generate_global_excel(assets_list):
    df_data = []
    for asset in assets_list:
        if not asset.get('images'):
            df_data.append({
                "Asset_ID": asset['id'],
                "Worker": asset['worker_name'],
                "Timestamp": asset['timestamp'],
                "Status": asset['status'],
                "Category": asset['asset_class'],
                "Voltage": asset['voltage'],
                "Component": "None",
                "Confidence": "",
                "Manual_Edit": "",
            })
            continue
        for i, img in enumerate(asset['images']):
            for det in img.get('detections', []):
                df_data.append({
                    "Asset_ID": asset['id'],
                    "Worker": asset['worker_name'],
                    "Timestamp": asset['timestamp'],
                    "Status": asset['status'],
                    "Category": asset['asset_class'],
                    "Voltage": asset['voltage'],
                    "Frame_No": i + 1,
                    "Component": det.get('label', 'UNKNOWN'),
                    "Confidence": det.get('confidence', 1.0),
                    "Manual_Edit": det.get('manual', False),
                })
    if not df_data:
        df_data.append({"Info": "No data found"})
    df = pd.DataFrame(df_data)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Global_Detection_Log')
    buffer.seek(0)
    return buffer



def register_admin_routes(app):
    @app.route('/admin')
    @admin_required
    def admin_dashboard():
        return render_template('admin.html')

    @app.route('/api/admin/users', methods=['GET'])
    @admin_required
    def get_all_users():
        conn = get_db_connection()
        users = conn.execute('SELECT id, username, role FROM users').fetchall()
        conn.close()
        return jsonify([dict(user) for user in users])

    @app.route('/api/admin/users/<username>', methods=['DELETE'])
    @admin_required
    def delete_user(username):
        if username == session.get('user'):
            return jsonify({'status': 'error', 'message': 'Cannot delete active session user'}), 400
        conn = get_db_connection()
        cursor = conn.execute('DELETE FROM users WHERE username = ?', (username,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        if deleted:
            log_activity(session.get('user', 'admin'), 'delete_user', f'Deleted user: {username}')
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'User not found'}), 404

    @app.route('/admin/export/global/excel')
    @admin_required
    def export_global_excel():
        conn = get_db_connection()
        assets = conn.execute('SELECT * FROM assets ORDER BY timestamp DESC').fetchall()
        asset_images = conn.execute('SELECT * FROM asset_images').fetchall()
        conn.close()
        image_map = {}
        for img in asset_images:
            asset_id = img['asset_id']
            image_map.setdefault(asset_id, [])
            image_map[asset_id].append(hydrate_asset_image_record(img))
        assets_list = []
        for asset in assets:
            asset_dict = dict(asset)
            asset_dict['images'] = image_map.get(asset['id'], [])
            assets_list.append(asset_dict)
        filename = f"Global_Inspection_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        return send_file(generate_global_excel(assets_list), download_name=filename, as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    @app.route('/admin/export/global/pdf')
    @admin_required
    def export_global_pdf():
        conn = get_db_connection()
        assets = conn.execute('SELECT * FROM assets ORDER BY timestamp DESC').fetchall()
        conn.close()
        filename = f"Global_Inspection_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        return send_file(generate_global_pdf([dict(asset) for asset in assets]), download_name=filename, as_attachment=True, mimetype='application/pdf')

    @app.route('/admin/asset/<asset_id>')
    @admin_required
    def view_asset(asset_id):
        conn = get_db_connection()
        asset_row = conn.execute('SELECT * FROM assets WHERE id = ?', (asset_id,)).fetchone()
        if not asset_row:
            conn.close()
            return 'Asset not found', 404
        image_rows = conn.execute('SELECT * FROM asset_images WHERE asset_id = ?', (asset_id,)).fetchall()

        # Fallback: if no asset_images rows, check uploaded_images by related_asset_id
        if not image_rows:
            from db import bytes_to_b64
            fallback_rows = conn.execute(
                'SELECT * FROM uploaded_images WHERE related_asset_id = ?', (asset_id,)
            ).fetchall()
            for fb_row in fallback_rows:
                fb = dict(fb_row)
                if fb.get('image_data'):
                    image_rows.append({
                        'id': fb['id'],
                        'asset_id': asset_id,
                        'image_b64': bytes_to_b64(fb['image_data']),
                        'detections': fb.get('detections', '[]'),
                        'pole_angle': 0.0,
                        'uploaded_image_id': fb['id'],
                    })

        conn.close()
        asset = dict(asset_row)
        images = []
        total_detections = 0
        for row in image_rows:
            if isinstance(row, dict):
                img_dict = row
                if isinstance(img_dict.get('detections'), str):
                    img_dict['detections'] = parse_db_json(img_dict['detections'])
                elif not img_dict.get('detections'):
                    img_dict['detections'] = []
            else:
                img_dict = hydrate_asset_image_record(row)
            total_detections += len(img_dict.get('detections', []))
            images.append(img_dict)
        asset['images'] = images
        asset['total_count'] = total_detections
        return render_template('asset_detail.html', asset=asset)

    @app.route('/admin/logs')
    @admin_required
    def audit_logs():
        conn = get_db_connection()
        logs = conn.execute('SELECT * FROM activity_logs ORDER BY timestamp DESC LIMIT 100').fetchall()
        conn.close()
        return render_template('audit_logs.html', logs=logs)

    @app.route('/admin/asset/pdf/<asset_id>')
    @admin_required
    def export_asset_pdf(asset_id):
        conn = get_db_connection()
        asset_row = conn.execute('SELECT * FROM assets WHERE id = ?', (asset_id,)).fetchone()
        if not asset_row:
            conn.close()
            return 'Asset not found', 404
        image_rows = conn.execute('SELECT * FROM asset_images WHERE asset_id = ?', (asset_id,)).fetchall()
        conn.close()
        asset_data = dict(asset_row)
        asset_data['images'] = [hydrate_asset_image_record(row) for row in image_rows]
        return send_file(generate_asset_pdf(asset_data), download_name=f'Inspection_Report_{asset_id[:8]}.pdf', as_attachment=True, mimetype='application/pdf')

    @app.route('/admin/asset/excel/<asset_id>')
    @admin_required
    def export_asset_excel(asset_id):
        conn = get_db_connection()
        asset_row = conn.execute('SELECT * FROM assets WHERE id = ?', (asset_id,)).fetchone()
        if not asset_row:
            conn.close()
            return 'Asset not found', 404
        image_rows = conn.execute('SELECT * FROM asset_images WHERE asset_id = ?', (asset_id,)).fetchall()
        conn.close()
        asset_data = dict(asset_row)
        asset_data['images'] = [hydrate_asset_image_record(row) for row in image_rows]
        return send_file(generate_asset_excel(asset_data), download_name=f'Detection_Log_{asset_id[:8]}.xlsx', as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    @app.route('/api/delete_asset/<asset_id>', methods=['DELETE'])
    @admin_required
    def delete_asset(asset_id):
        conn = get_db_connection()
        try:
            conn.execute('DELETE FROM asset_images WHERE asset_id = ?', (asset_id,))
            conn.execute('DELETE FROM uploaded_images WHERE related_asset_id = ?', (asset_id,))
            conn.execute('DELETE FROM pole_surveys WHERE id = ?', (asset_id,))
            conn.execute('DELETE FROM assets WHERE id = ?', (asset_id,))
            conn.commit()
            log_activity(session['user'], 'asset_delete_full', f'Permanently deleted Asset: {asset_id}')
            return jsonify({'status': 'success'})
        except Exception as exc:
            conn.rollback()
            return jsonify({'status': 'error', 'message': str(exc)}), 500
        finally:
            conn.close()

    @app.route('/api/update_asset_detections', methods=['POST'])
    @admin_required
    def update_asset_detections():
        data = request.json
        conn = get_db_connection()
        try:
            image_rows = conn.execute('SELECT id FROM asset_images WHERE asset_id = ? ORDER BY id ASC', (data.get('asset_id'),)).fetchall()
            for update in data.get('updates', []):
                if update['index'] < len(image_rows):
                    img_db_id = image_rows[update['index']]['id']
                    conn.execute('UPDATE asset_images SET detections = ? WHERE id = ?', (json.dumps(update['detections']), img_db_id))
            conn.commit()
            log_activity(session['user'], 'asset_annotation_edit', f"Modified annotations for Asset: {data.get('asset_id')}")
            return jsonify({'status': 'success'})
        except Exception as exc:
            conn.rollback()
            return jsonify({'status': 'error', 'message': str(exc)}), 500
        finally:
            conn.close()

    @app.route('/api/delete_asset_image', methods=['POST'])
    @admin_required
    def delete_asset_image():
        data = request.json
        conn = get_db_connection()
        try:
            conn.execute('DELETE FROM asset_images WHERE id = ?', (data.get('image_id'),))
            conn.commit()
            log_activity(session['user'], 'asset_image_delete', f"Deleted Image ID: {data.get('image_id')}")
            return jsonify({'status': 'success'})
        except Exception as exc:
            conn.rollback()
            return jsonify({'status': 'error', 'message': str(exc)}), 500
        finally:
            conn.close()

    @app.route('/api/get_asset_history/<asset_id>')
    @admin_required
    def get_asset_history(asset_id):
        conn = get_db_connection()
        try:
            asset = conn.execute('SELECT worker_name, timestamp FROM assets WHERE id = ?', (asset_id,)).fetchone()
            if not asset:
                return jsonify({'worker_name': 'Unknown', 'submitted_at': '', 'images': [], 'logs': []})
            image_rows = conn.execute('SELECT * FROM asset_images WHERE asset_id = ?', (asset_id,)).fetchall()
            images = [hydrate_asset_image_record(row) for row in image_rows]

            # Fallback: if no asset_images rows, check uploaded_images by related_asset_id
            if not images:
                from db import bytes_to_b64
                fallback_rows = conn.execute(
                    'SELECT * FROM uploaded_images WHERE related_asset_id = ?', (asset_id,)
                ).fetchall()
                for fb_row in fallback_rows:
                    fb = dict(fb_row)
                    if fb.get('image_data'):
                        images.append({
                            'id': fb['id'],
                            'asset_id': asset_id,
                            'image_b64': bytes_to_b64(fb['image_data']),
                            'detections': parse_db_json(fb.get('detections', '[]')),
                            'pole_angle': 0.0,
                            'uploaded_image_id': fb['id'],
                        })

            logs = conn.execute('SELECT user_name, action, details, timestamp FROM activity_logs WHERE details LIKE ? ORDER BY timestamp ASC', (f'%{asset_id}%',)).fetchall()
            return jsonify({'worker_name': asset['worker_name'], 'submitted_at': asset['timestamp'], 'images': images, 'logs': [dict(log) for log in logs]})
        except Exception as exc:
            print(f'[app] Error fetching history for asset {asset_id}: {exc}')
            return jsonify({'status': 'error', 'message': str(exc)}), 500
        finally:
            conn.close()

    @app.route('/api/training_stats')
    @login_required
    def training_stats():
        try:
            return jsonify(build_training_dashboard_stats(get_training_stats()))
        except Exception as exc:
            print(f'[app] Training stats unavailable: {exc}')
            return jsonify({'total_samples': 0, 'total_classes': 0, 'total_annotations': 0, 'avg_confidence': 0, 'by_class': {}, 'class_confidence': {}, 'models': [], 'datasets': [], 'status': 'unavailable'})

    @app.route('/api/training_export/<asset_id>', methods=['POST'])
    @admin_required
    def manual_training_export(asset_id):
        try:
            return jsonify({'status': 'success', **export_asset_to_training(asset_id, approved_by=session['user'])})
        except Exception as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 500
