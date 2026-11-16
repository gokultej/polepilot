from functools import wraps

from flask import redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from db import get_db_connection, log_activity


def register_auth_routes(app):
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            conn = get_db_connection()
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            conn.close()
            if user and check_password_hash(user['password'], password):
                session.clear()
                session['user'] = user['username']
                session['role'] = user['role']
                log_activity(username, 'login', f"Role: {user['role']}")
                if user['role'] == 'admin':
                    return redirect(url_for('admin_dashboard'))
                return redirect(url_for('home'))
            return render_template('login.html', error='Invalid credentials')
        return render_template('login.html')

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')
            if not username or not password:
                return render_template('signup.html', error='All fields are required')
            if password != confirm_password:
                return render_template('signup.html', error='Passwords do not match')
            if len(password) < 12:
                return render_template('signup.html', error='Use a password with at least 12 characters')
            conn = get_db_connection()
            existing = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            if existing:
                conn.close()
                return render_template('signup.html', error='Username already exists')
            hashed_pw = generate_password_hash(password)
            conn.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, hashed_pw, 'user'))
            conn.commit()
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            conn.close()
            session.clear()
            session['user'] = user['username']
            session['role'] = user['role']
            log_activity(username, 'signup', 'New user registered via web UI')
            return redirect(url_for('home'))
        return render_template('signup.html')

    @app.route('/logout')
    def logout():
        session.clear()
        return redirect(url_for('login'))

    @app.route('/demo_login')
    def demo_login():
        return redirect('/demo')

    @app.route('/demo_logout')
    def demo_logout():
        session.clear()
        return redirect(url_for('login'))


def login_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return func(*args, **kwargs)

    return decorated_function


def admin_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or session['role'] != 'admin':
            return redirect(url_for('home'))
        return func(*args, **kwargs)

    return decorated_function
