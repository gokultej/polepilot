"""Explicit administrator provisioning; no default passwords are seeded."""
import argparse
import getpass

from werkzeug.security import generate_password_hash
from db import get_db_connection, init_postgres_db


def main():
    parser = argparse.ArgumentParser(description='Provision a PolePilot administrator')
    parser.add_argument('command', choices=['create-admin'])
    parser.add_argument('--username', default='admin')
    args = parser.parse_args()
    password = getpass.getpass('New administrator password (12+ characters): ')
    if len(password) < 12 or password != getpass.getpass('Confirm password: '):
        parser.error('Passwords must match and contain at least 12 characters.')
    init_postgres_db(seed_defaults=False)
    conn = get_db_connection()
    try:
        if conn.execute('SELECT id FROM users WHERE username=?', (args.username,)).fetchone():
            parser.error('That username already exists. Choose a new administrator name.')
        conn.execute('INSERT INTO users (username,password,role) VALUES (?,?,?)', (args.username, generate_password_hash(password), 'admin'))
        conn.commit()
        print('Administrator created. Sign in through /login over HTTPS.')
    finally:
        conn.close()


if __name__ == '__main__':
    main()
