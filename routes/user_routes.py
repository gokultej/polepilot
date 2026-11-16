from flask import render_template, session

from routes.auth_routes import login_required


def register_user_routes(app):
    @app.route('/workspace')
    @login_required
    def home():
        return render_template('index.html', role=session.get('role', 'user'))
