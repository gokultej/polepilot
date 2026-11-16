"""PolePilot public experience; enable the PostgreSQL workspace explicitly."""
import hmac
import os
import secrets
from datetime import timedelta
from pathlib import Path

from flask import Flask, jsonify, redirect, request, session
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from config import APP_SECRET_KEY, THRESHOLD_POLE


def enabled(name, default=False):
    return os.getenv(name, str(default)).lower() in ('true', '1', 'yes')


def create_app(test_config=None):
    app = Flask(__name__)
    production = os.getenv('APP_ENV', 'development') == 'production'
    if production and (APP_SECRET_KEY == 'dev-secret-change-me' or APP_SECRET_KEY.startswith('replace-') or len(APP_SECRET_KEY) < 32):
        raise RuntimeError('Set a random APP_SECRET_KEY of at least 32 characters for production.')
    app.config.update(
        SECRET_KEY=APP_SECRET_KEY if APP_SECRET_KEY != 'dev-secret-change-me' else secrets.token_hex(32),
        PRODUCT_NAME=os.getenv('PRODUCT_NAME', 'PolePilot'),
        WORKSPACE_ENABLED=enabled('WORKSPACE_ENABLED'),
        LIVE_INFERENCE_ENABLED=enabled('LIVE_INFERENCE_ENABLED', True),
        EVALUATION_DATA_DIR=os.getenv('EVALUATION_DATA_DIR', str(Path(__file__).parent / '.data')),
        MAX_CONTENT_LENGTH=110 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax', SESSION_COOKIE_SECURE=production,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=24),
    )
    if test_config:
        app.config.update(test_config)
    if enabled('TRUST_PROXY'):
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    @app.before_request
    def request_security():
        session.permanent = True
        session.setdefault('csrf_token', secrets.token_urlsafe(32))
        if request.path == '/api/evaluation/jobs' and request.content_length and request.content_length > 11 * 1024 * 1024:
            return jsonify(error='Images must be 10 MB or smaller.'), 413
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token', '')
            if not hmac.compare_digest(token, session['csrf_token']):
                return jsonify(error='Your session token is missing or expired. Reload the page and retry.'), 403

    @app.context_processor
    def public_context():
        return {'product_name': app.config['PRODUCT_NAME'], 'workspace_enabled': app.config['WORKSPACE_ENABLED'], 'csrf_token': session.get('csrf_token', ''), 'pole_min_confidence_percent': int(THRESHOLD_POLE * 100)}

    @app.after_request
    def security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        if request.path.startswith('/api/evaluation') or response.mimetype == 'text/html':
            response.headers['Cache-Control'] = 'no-store'
        if production:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000'
        return response

    @app.errorhandler(HTTPException)
    def http_error(exc):
        if request.path.startswith('/api/'):
            return jsonify(error=exc.description), exc.code
        return exc

    from evaluation.routes import register_evaluation_routes
    register_evaluation_routes(app)

    @app.get('/health/live')
    def health_live():
        return jsonify(status='ok', product=app.config['PRODUCT_NAME'])

    @app.get('/health/ready')
    def health_ready():
        try:
            with app.extensions['evaluation_store'].connection() as conn:
                conn.execute('SELECT 1')
            if app.config['WORKSPACE_ENABLED']:
                from db import get_db_connection
                conn = get_db_connection()
                try:
                    conn.execute('SELECT 1')
                finally:
                    conn.close()
        except Exception:
            app.logger.exception('Readiness check failed')
            return jsonify(status='unavailable'), 503
        return jsonify(status='ready', workspace_enabled=app.config['WORKSPACE_ENABLED'])

    if app.config['WORKSPACE_ENABLED']:
        from db import init_postgres_db, register_db_routes
        from routes.admin_routes import register_admin_routes
        from routes.auth_routes import register_auth_routes
        from routes.image_routes import register_image_routes
        from routes.pole_vegetation_routes import register_pole_vegetation_routes
        from routes.tree_routes import register_tree_routes
        from routes.user_routes import register_user_routes
        from routes.video_routes import register_video_routes
        from routes.survey_routes import register_survey_routes
        init_postgres_db(seed_defaults=False)
        for register in (register_auth_routes, register_user_routes, register_image_routes, register_tree_routes, register_pole_vegetation_routes, register_video_routes, register_survey_routes, register_db_routes, register_admin_routes):
            register(app)
    else:
        @app.get('/workspace')
        @app.get('/login')
        @app.get('/demo_login')
        def workspace_unavailable():
            return redirect('/demo')
    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.getenv('PORT', '5002')), debug=False)
