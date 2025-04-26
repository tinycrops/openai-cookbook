import sqlite3
import threading
import json
from datetime import datetime

class SignalRecorder:
    """
    A simple signal recorder using SQLite to record important signals for agentic systems.
    """
    def __init__(self, db_path='signals.db'):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                signal_name TEXT NOT NULL,
                signal_value REAL,
                metadata TEXT,
                image BLOB
            )
        """)
        self._conn.commit()

    def record_signal(self, name, value=None, metadata=None):
        """
        Record a signal with a name, numeric value, and optional metadata dict.
        """
        ts = datetime.utcnow().isoformat() + 'Z'
        meta_json = json.dumps(metadata) if metadata is not None else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO signals (timestamp, signal_name, signal_value, metadata) VALUES (?, ?, ?, ?)",
                (ts, name, value, meta_json)
            )
            self._conn.commit()

    def record_image_signal(self, name, image_bytes, metadata=None):
        """
        Record a signal with a name, image bytes, and optional metadata dict.
        """
        ts = datetime.utcnow().isoformat() + 'Z'
        meta_json = json.dumps(metadata) if metadata is not None else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO signals (timestamp, signal_name, signal_value, metadata, image) VALUES (?, ?, ?, ?, ?)",
                (ts, name, None, meta_json, image_bytes)
            )
            self._conn.commit()

    def fetch_signals(self, name=None, limit=100):
        """
        Fetch recent signals. Optionally filter by name.
        """
        query = "SELECT id, timestamp, signal_name, signal_value, metadata FROM signals"
        params = []
        if name:
            query += " WHERE signal_name = ?"
            params.append(name)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
        results = []
        for id_, ts, sname, sval, meta in rows:
            results.append({
                'id': id_,
                'timestamp': ts,
                'signal_name': sname,
                'signal_value': sval,
                'metadata': json.loads(meta) if meta else None
            })
        return results

    def close(self):
        """Close the database connection."""
        with self._lock:
            self._conn.close()