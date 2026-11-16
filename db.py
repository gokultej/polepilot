import json
import base64
import uuid
from datetime import datetime

from werkzeug.security import generate_password_hash
import psycopg2
from psycopg2.extras import RealDictCursor

from config import PG_DB, PG_HOST, PG_PASS, PG_PORT, PG_USER


# Bootstrap is explicit; never ship predictable administrator passwords.
DEFAULT_USERS = []

LEGACY_DEFAULT_USERNAMES = {
    "admin 1": "admin",
    "user 1": "user",
}

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS assets (
        id TEXT PRIMARY KEY,
        worker_name TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        timestamp TEXT NOT NULL,
        asset_class TEXT,
        voltage TEXT,
        reason TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS uploaded_images (
        id TEXT PRIMARY KEY,
        original_filename TEXT,
        mime_type TEXT,
        image_data BYTEA NOT NULL,
        related_asset_id TEXT REFERENCES assets(id) ON DELETE SET NULL,
        detections TEXT,
        created_at TEXT NOT NULL,
        source_path TEXT UNIQUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS asset_images (
        id SERIAL PRIMARY KEY,
        asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
        image_b64 TEXT NOT NULL,
        detections TEXT NOT NULL,
        pole_angle DOUBLE PRECISION DEFAULT 0.0,
        uploaded_image_id TEXT REFERENCES uploaded_images(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS activity_logs (
        id SERIAL PRIMARY KEY,
        user_name TEXT NOT NULL,
        action TEXT NOT NULL,
        details TEXT,
        timestamp TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS drafts (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        data TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_samples (
        id TEXT PRIMARY KEY,
        asset_id TEXT,
        image_file TEXT,
        label_file TEXT,
        class_counts TEXT,
        approved_by TEXT,
        timestamp TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_runs (
        id SERIAL PRIMARY KEY,
        triggered_at TEXT,
        sample_count INTEGER,
        status TEXT DEFAULT 'queued',
        result TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS uploaded_videos (
        id TEXT PRIMARY KEY,
        original_filename TEXT,
        mime_type TEXT,
        video_data BYTEA NOT NULL,
        created_at TEXT NOT NULL,
        source_path TEXT UNIQUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pole_vegetation_reports (
        id TEXT PRIMARY KEY,
        pole_id TEXT,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        capture_date TEXT,
        season TEXT,
        created_at TEXT NOT NULL,
        original_image_path TEXT,
        original_uploaded_image_id TEXT REFERENCES uploaded_images(id) ON DELETE SET NULL,
        azure_context_json TEXT,
        report_json TEXT NOT NULL,
        user_selected_tree_id TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pole_surveys (
        id TEXT PRIMARY KEY,
        pole_location_number TEXT,
        schema_id TEXT,
        title TEXT,
        survey_data_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
]


class DBConn:
    """Small adapter that preserves the app's existing ? placeholders."""

    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=()):
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql.replace("?", "%s"), params)
        return cur

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()


def get_db_connection():
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DB,
        user=PG_USER,
        password=PG_PASS,
        connect_timeout=5,
    )
    return DBConn(conn)


def seed_default_users(conn):
    for username, password, role in DEFAULT_USERS:
        conn.execute(
            """
            INSERT INTO users (username, password, role)
            VALUES (?, ?, ?)
            ON CONFLICT (username) DO NOTHING
            """,
            (username, generate_password_hash(password), role),
        )


def migrate_legacy_default_usernames(conn):
    for old_username, new_username in LEGACY_DEFAULT_USERNAMES.items():
        old_user = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (old_username,),
        ).fetchone()
        if not old_user:
            continue

        new_user = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (new_username,),
        ).fetchone()
        if new_user:
            continue

        conn.execute(
            "UPDATE users SET username = ? WHERE username = ?",
            (new_username, old_username),
        )


def init_postgres_db(seed_defaults=True):
    conn = get_db_connection()
    try:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
        conn.execute(
            "ALTER TABLE asset_images ADD COLUMN IF NOT EXISTS uploaded_image_id TEXT REFERENCES uploaded_images(id) ON DELETE SET NULL"
        )
        conn.execute(
            "ALTER TABLE uploaded_videos ADD COLUMN IF NOT EXISTS source_path TEXT UNIQUE"
        )
        if seed_defaults:
            migrate_legacy_default_usernames(conn)
            seed_default_users(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def clean_b64(b64_str):
    if not b64_str:
        return ""
    b64_str = str(b64_str).strip()
    if "base64," in b64_str:
        b64_str = b64_str.split("base64,")[-1]
    elif "," in b64_str:
        b64_str = b64_str.split(",")[-1]
    b64_str = "".join(b64_str.split())
    b64_str = b64_str.replace("-", "+").replace("_", "/")
    missing_padding = len(b64_str) % 4
    if missing_padding == 1:
        b64_str = b64_str[:-1]
    elif missing_padding > 1:
        b64_str += "=" * (4 - missing_padding)
    return b64_str


def b64_to_bytes(b64_str):
    normalized = clean_b64(b64_str)
    if not normalized:
        return b""
    return base64.b64decode(normalized)


def bytes_to_b64(raw_bytes):
    if not raw_bytes:
        return ""
    return base64.b64encode(bytes(raw_bytes)).decode("utf-8")


def parse_db_json(field_data):
    if not field_data:
        return []
    if isinstance(field_data, str):
        try:
            return json.loads(field_data)
        except Exception:
            return []
    return field_data


def save_uploaded_image(original_filename, mime_type, image_bytes, related_asset_id=None, detections=None, source_path=None):
    image_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO uploaded_images (id, original_filename, mime_type, image_data, related_asset_id, detections, created_at, source_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                original_filename,
                mime_type,
                psycopg2.Binary(image_bytes),
                related_asset_id,
                detections,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                source_path,
            ),
        )
        conn.commit()
        return image_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_uploaded_image(image_id):
    conn = get_db_connection()
    try:
        return conn.execute(
            """
            SELECT id, original_filename, mime_type, image_data, related_asset_id, detections, created_at, source_path
            FROM uploaded_images
            WHERE id = ?
            """,
            (image_id,),
        ).fetchone()
    finally:
        conn.close()


def update_uploaded_image(image_id, related_asset_id=None, detections=None):
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT id, image_data, mime_type, original_filename FROM uploaded_images WHERE id = ?",
            (image_id,),
        ).fetchone()
        if not row:
            return None
        conn.execute(
            """
            UPDATE uploaded_images
            SET related_asset_id = COALESCE(?, related_asset_id),
                detections = COALESCE(?, detections)
            WHERE id = ?
            """,
            (related_asset_id, detections, image_id),
        )
        conn.commit()
        return row
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_uploaded_video(original_filename, mime_type, video_bytes):
    video_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO uploaded_videos (id, original_filename, mime_type, video_data, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                video_id,
                original_filename,
                mime_type,
                psycopg2.Binary(video_bytes),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return video_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_uploaded_video(video_id):
    conn = get_db_connection()
    try:
        return conn.execute(
            """
            SELECT id, original_filename, mime_type, video_data, created_at
            FROM uploaded_videos
            WHERE id = ?
            """,
            (video_id,),
        ).fetchone()
    finally:
        conn.close()


def hydrate_asset_image_record(row):
    image = dict(row)
    uploaded_image_id = image.get("uploaded_image_id")
    if uploaded_image_id:
        uploaded_row = get_uploaded_image(uploaded_image_id)
        if uploaded_row and uploaded_row.get("image_data"):
            image["image_b64"] = bytes_to_b64(uploaded_row["image_data"])
    if image.get("detections"):
        image["detections"] = parse_db_json(image["detections"])
    else:
        image["detections"] = []
    return image


def log_activity(user, action, details=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DB_LOG] user={user} ({type(user)}), action={action} ({type(action)}), details={details} ({type(details)})")
    safe_user = str(user) if user is not None else "system"
    safe_action = str(action) if action is not None else "unknown"
    safe_details = json.dumps(details) if isinstance(details, (dict, list)) else (str(details) if details is not None else "")
    conn = get_db_connection()
    try:
        conn.execute(
            'INSERT INTO activity_logs (user_name, action, details, timestamp) VALUES (?, ?, ?, ?)',
            (safe_user, safe_action, safe_details, timestamp),
        )
        conn.commit()
    except Exception as exc:
        print(f"[DB_ERROR] log_activity failed: {exc}")
    finally:
        conn.close()


def register_db_routes(app):
    from flask import jsonify, request, session
    from routes.auth_routes import login_required, admin_required
    from training_pipeline import export_asset_to_training

    @app.route('/api/save_asset', methods=['POST'])
    @login_required
    def save_asset():
        data = request.json
        asset_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        worker_name = session['user']
        master = data.get('master') if isinstance(data.get('master'), dict) else {}
        conn = get_db_connection()
        try:
            a_class = master.get('final_class')
            a_volt = master.get('voltage')
            a_reason = master.get('reason')
            if isinstance(a_class, (dict, list)):
                a_class = json.dumps(a_class)
            if isinstance(a_volt, (dict, list)):
                a_volt = json.dumps(a_volt)
            if isinstance(a_reason, (dict, list)):
                a_reason = json.dumps(a_reason)
            conn.execute(
                """
                INSERT INTO assets (id, worker_name, status, timestamp, asset_class, voltage, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(asset_id),
                    str(worker_name),
                    'pending',
                    str(timestamp),
                    str(a_class) if a_class is not None else None,
                    str(a_volt) if a_volt is not None else None,
                    str(a_reason) if a_reason is not None else None,
                ),
            )

            for index, img_data in enumerate(data['images']):
                detections = img_data.get('detections', [])
                det_str = json.dumps(detections) if not isinstance(detections, str) else detections
                uploaded_image_id = img_data.get('uploaded_image_id')
                normalized_b64 = ''

                if uploaded_image_id:
                    uploaded_row = conn.execute(
                        'SELECT id, image_data FROM uploaded_images WHERE id = ?',
                        (uploaded_image_id,),
                    ).fetchone()
                    if uploaded_row:
                        normalized_b64 = bytes_to_b64(uploaded_row['image_data'])
                        conn.execute(
                            'UPDATE uploaded_images SET related_asset_id = ?, detections = ? WHERE id = ?',
                            (asset_id, det_str, uploaded_image_id),
                        )
                    else:
                        uploaded_image_id = None

                if not uploaded_image_id:
                    raw_b64 = str(img_data.get('image_b64', ''))
                    if not raw_b64 or raw_b64 == 'None':
                        continue
                    image_bytes = b64_to_bytes(raw_b64)
                    normalized_b64 = bytes_to_b64(image_bytes)
                    uploaded_image_id = str(uuid.uuid4())
                    conn.execute(
                        """
                        INSERT INTO uploaded_images (id, original_filename, mime_type, image_data, related_asset_id, detections, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            uploaded_image_id,
                            img_data.get('filename') or f'asset_{asset_id}_{index + 1}.jpg',
                            img_data.get('mime_type') or 'image/jpeg',
                            psycopg2.Binary(image_bytes),
                            asset_id,
                            det_str,
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        ),
                    )

                conn.execute(
                    """
                    INSERT INTO asset_images (asset_id, image_b64, detections, pole_angle, uploaded_image_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(asset_id),
                        normalized_b64,
                        det_str,
                        float(img_data.get('pole_angle', 0.0)),
                        uploaded_image_id,
                    ),
                )
            conn.commit()
            log_activity(worker_name, 'asset_submission', f'Asset: {asset_id}, Images: {len(data["images"])}')
        except Exception as exc:
            conn.rollback()
            return jsonify({'status': 'error', 'message': str(exc)}), 500
        finally:
            conn.close()
        return jsonify({'status': 'success', 'asset_id': asset_id})

    @app.route('/api/save_draft', methods=['POST'])
    @login_required
    def save_draft():
        data = request.get_json(silent=True) or {}
        draft_id = data.get('id')
        if not isinstance(draft_id, str) or not draft_id or len(draft_id) > 200:
            return jsonify({'error': 'A valid draft id is required'}), 400
        owned_id = session['user'] + ':' + draft_id
        content = data.get('data')
        if isinstance(content, (dict, list)):
            content = json.dumps(content)
        else:
            content = str(content) if content is not None else ''
        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO drafts (id, type, data, timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                type = EXCLUDED.type,
                data = EXCLUDED.data,
                timestamp = EXCLUDED.timestamp
            """,
            (owned_id, data.get('type', 'worker'), content, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        )
        conn.commit()
        conn.close()
        return jsonify({'status': 'success'})

    @app.route('/api/get_draft/<draft_id>')
    @login_required
    def get_draft(draft_id):
        conn = get_db_connection()
        draft = conn.execute('SELECT * FROM drafts WHERE id = ?', (session['user'] + ':' + draft_id,)).fetchone()
        conn.close()
        if draft:
            return jsonify({'status': 'success', 'data': draft['data']})
        return jsonify({'status': 'error', 'message': 'Draft not found'}), 404

    @app.route('/api/get_assets')
    @login_required
    def get_assets():
        status = request.args.get('status')
        conn = get_db_connection()
        query = """
            SELECT a.*, i.image_b64 as thumbnail, i.detections as detections, i.uploaded_image_id as uploaded_image_id
            FROM assets a
            LEFT JOIN asset_images i ON i.id = (
                SELECT id FROM asset_images WHERE asset_id = a.id LIMIT 1
            )
            WHERE 1=1
        """
        params = []
        if status:
            query += ' AND a.status = ?'
            params.append(status)
        if session['role'] != 'admin':
            query += ' AND a.worker_name = ?'
            params.append(session['user'])
        query += ' ORDER BY a.timestamp DESC'
        rows = conn.execute(query, params).fetchall()
        data = []
        for row in rows:
            item = dict(row)
            if item.get('uploaded_image_id'):
                item = hydrate_asset_image_record(item)
                if 'image_b64' in item:
                    item['thumbnail'] = item['image_b64']
            elif not item.get('thumbnail'):
                # Fallback: check uploaded_images directly by related_asset_id
                fallback_row = conn.execute(
                    'SELECT id, image_data FROM uploaded_images WHERE related_asset_id = ? LIMIT 1',
                    (item['id'],)
                ).fetchone()
                if fallback_row and fallback_row.get('image_data'):
                    item['thumbnail'] = bytes_to_b64(fallback_row['image_data'])
                    item['uploaded_image_id'] = fallback_row['id']
            if item.get('detections') and isinstance(item['detections'], str):
                item['detections'] = parse_db_json(item['detections'])
            data.append(item)
        conn.close()
        return jsonify(data)

    @app.route('/api/update_asset_status', methods=['POST'])
    @admin_required
    def update_asset_status():
        data = request.json
        asset_id = data.get('asset_id')
        status = data.get('status')
        if status not in ('pending', 'approved', 'rejected'):
            return jsonify({'error': 'Invalid asset status'}), 400
        conn = get_db_connection()
        conn.execute('UPDATE assets SET status = ? WHERE id = ?', (status, asset_id))
        conn.commit()
        conn.close()
        log_activity(session['user'], 'asset_status_update', f'Asset: {asset_id}, Status: {status}')
        if status == 'approved':
            try:
                result = export_asset_to_training(asset_id, approved_by=session['user'])
                log_activity(session['user'], 'training_data_exported', f"Asset: {asset_id} | Exported: {result['exported']} images | Classes: {result['classes']} | Pool: {result['total_pool']}")
            except Exception as exc:
                print(f'[app] Training export error: {exc}')
        return jsonify({'status': 'success'})
