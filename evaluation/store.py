"""Small durable job store. Every lookup is scoped to the signed session owner."""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class JobStore:
    def __init__(self, directory, retention_hours=24):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.retention = retention_hours * 3600
        self.path = self.directory / 'evaluation.sqlite3'
        with self.connection() as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, created REAL NOT NULL, status TEXT NOT NULL, source TEXT NOT NULL, result TEXT, error TEXT, review TEXT, image_path TEXT)')
            conn.execute('CREATE INDEX IF NOT EXISTS jobs_owner_created ON jobs(owner, created)')
            conn.execute("UPDATE jobs SET status='failed', error='Server restarted. Please run the inspection again.' WHERE status IN ('queued','running')")

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def cleanup(self):
        with self.connection() as conn:
            rows = conn.execute("SELECT image_path FROM jobs WHERE created < ? AND status NOT IN ('queued','running')", (time.time() - self.retention,)).fetchall()
            conn.execute("DELETE FROM jobs WHERE created < ? AND status NOT IN ('queued','running')", (time.time() - self.retention,))
        for row in rows:
            if row['image_path']:
                (self.directory / row['image_path']).unlink(missing_ok=True)

    def add(self, job_id, owner, source, image_path=None, result=None):
        self.cleanup()
        with self.connection() as conn:
            # Bound per-session storage for repeated replay requests as well as uploads.
            count = conn.execute('SELECT COUNT(*) FROM jobs WHERE owner=?', (owner,)).fetchone()[0]
            if count >= 30:
                raise ValueError('Your demo has 30 inspections. Reset it before starting another.')
            conn.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)', (job_id, owner, time.time(), 'done' if result else 'queued', source, json.dumps(result) if result else None, None, None, image_path))

    def get(self, job_id, owner):
        with self.connection() as conn:
            row = conn.execute('SELECT * FROM jobs WHERE id=? AND owner=? AND created>=?', (job_id, owner, time.time() - self.retention)).fetchone()
        if not row:
            return None
        data = dict(row)
        for key in ('result', 'review'):
            data[key] = json.loads(data[key]) if data[key] else None
        return data

    def update(self, job_id, **changes):
        allowed = {'status', 'result', 'error', 'review'}
        if not changes or set(changes) - allowed:
            raise ValueError('Invalid job update')
        values = [json.dumps(value) if key in ('result', 'review') else value for key, value in changes.items()]
        with self.connection() as conn:
            conn.execute('UPDATE jobs SET ' + ', '.join(key + '=?' for key in changes) + ' WHERE id=?', (*values, job_id))

    def live_count(self, owner):
        with self.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM jobs WHERE owner=? AND created>? AND source='live'", (owner, time.time() - 900)).fetchone()[0]

    def reset(self, owner):
        with self.connection() as conn:
            if conn.execute("SELECT 1 FROM jobs WHERE owner=? AND status IN ('queued','running')", (owner,)).fetchone():
                raise ValueError('Wait for the current inspection to finish before resetting.')
            rows = conn.execute('SELECT image_path FROM jobs WHERE owner=?', (owner,)).fetchall()
            conn.execute('DELETE FROM jobs WHERE owner=?', (owner,))
        for row in rows:
            if row['image_path']:
                (self.directory / row['image_path']).unlink(missing_ok=True)
