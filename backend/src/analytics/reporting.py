"""
Analytics and Reporting Module.

Generates violation statistics, trends, and searchable records.
"""

import json
import os
import sqlite3
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

import pandas as pd

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


class DatabaseConnection:
    def __init__(self, is_postgres: bool, conn):
        self.is_postgres = is_postgres
        self.conn = conn

    def execute(self, query: str, params=None):
        if self.is_postgres:
            if params is not None:
                query = query.replace("?", "%s")
            cur = self.conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, params)
            return cur
        else:
            if params is not None:
                return self.conn.execute(query, params)
            else:
                return self.conn.execute(query)

    def executemany(self, query: str, params_list: list):
        if self.is_postgres:
            query = query.replace("?", "%s")
            cur = self.conn.cursor()
            cur.executemany(query, params_list)
            return cur
        else:
            return self.conn.executemany(query, params_list)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from ..evidence.generator import EvidencePackage
from ..evidence import integrity


class ViolationDatabase:
    """Concurrency-safe, SQLite/PostgreSQL-backed violation record store.

    Replaces the previous whole-file JSON rewrite (which lost updates under
    parallel writers and rewrote the entire file on every insert). SQLite in
    WAL mode or PostgreSQL gives us:
      • atomic, durable appends that scale to large volumes,
      • safe concurrent access from multiple worker threads/processes,
      • indexed search instead of full in-memory scans.

    The public interface (records, add_record, add_records, search,
    get_dataframe, _load) is preserved so existing callers are unaffected.

    Every inserted record is sealed into a tamper-evident hash chain
    (see src/evidence/integrity.py): each row stores the previous row's
    record_hash, so any later edit/deletion is detectable.
    """

    def __init__(self, db_path: str = "data/violations_db.json"):
        # Check if DATABASE_URL is set (PostgreSQL mode)
        self.database_url = os.environ.get("DATABASE_URL")
        self.is_postgres = False
        if self.database_url and (self.database_url.startswith("postgres://") or self.database_url.startswith("postgresql://")):
            self.is_postgres = True

        if self.is_postgres:
            self.legacy_json_path = None
            self.db_path = None
        else:
            # Accept the historical ".json" path but store in a sibling SQLite file.
            p = Path(db_path)
            if p.suffix.lower() == ".json":
                self.legacy_json_path = p
                self.db_path = p.with_suffix(".sqlite")
            else:
                self.legacy_json_path = None
                self.db_path = p
        self.records = []
        self._lock = threading.Lock()
        self._conn = None
        self._init_db()
        if not self.is_postgres:
            self._migrate_legacy_json()
        self._load()

    def _connect(self) -> DatabaseConnection:
        if self.is_postgres:
            if not HAS_PSYCOPG2:
                raise ImportError("psycopg2-binary package is required for PostgreSQL connection.")
            # Always open a new connection for postgres to avoid stale sockets in serverless
            conn = psycopg2.connect(self.database_url)
            return DatabaseConnection(True, conn)
        else:
            if self._conn is None:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(
                    str(self.db_path), check_same_thread=False, timeout=30.0
                )
                self._conn.row_factory = sqlite3.Row
                try:
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA synchronous=NORMAL")
                except sqlite3.OperationalError:
                    pass
            return DatabaseConnection(False, self._conn)

    def _init_db(self):
        conn = self._connect()
        try:
            with self._lock:
                if self.is_postgres:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS violations (
                            id            SERIAL PRIMARY KEY,
                            violation_id  TEXT,
                            timestamp     TEXT,
                            violation_type TEXT,
                            confidence    REAL,
                            severity      TEXT,
                            vehicle_plate TEXT,
                            content_hash  TEXT,
                            prev_hash     TEXT,
                            record_hash   TEXT,
                            sealed_at     TEXT,
                            record_json   TEXT NOT NULL
                        )
                        """
                    )
                else:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS violations (
                            id            INTEGER PRIMARY KEY AUTOINCREMENT,
                            violation_id  TEXT,
                            timestamp     TEXT,
                            violation_type TEXT,
                            confidence    REAL,
                            severity      TEXT,
                            vehicle_plate TEXT,
                            content_hash  TEXT,
                            prev_hash     TEXT,
                            record_hash   TEXT,
                            sealed_at     TEXT,
                            record_json   TEXT NOT NULL
                        )
                        """
                    )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_vtype ON violations(violation_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_plate ON violations(vehicle_plate)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON violations(timestamp)")
                conn.commit()
        finally:
            conn.close()

    def _migrate_legacy_json(self):
        """One-time import of an existing violations_db.json into SQLite."""
        if not self.legacy_json_path or not self.legacy_json_path.exists():
            return
        conn = self._connect()
        try:
            with self._lock:
                cur = conn.execute("SELECT COUNT(*) AS n FROM violations")
                if cur.fetchone()["n"] > 0:
                    return  # already populated; don't double-import
            try:
                old = json.loads(self.legacy_json_path.read_text())
            except (json.JSONDecodeError, ValueError, OSError):
                return
            if isinstance(old, list) and old:
                self._insert_records(old)
                # Keep the JSON as a backup; rename so we don't re-import.
                try:
                    self.legacy_json_path.rename(self.legacy_json_path.with_suffix(".json.imported"))
                except OSError:
                    pass
        finally:
            conn.close()

    # ── chain helpers ──
    def _last_record_hash(self, conn) -> str:
        cur = conn.execute("SELECT record_hash FROM violations ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        return row["record_hash"] if row and row["record_hash"] else integrity.GENESIS_HASH

    def _insert_records(self, record_dicts: list):
        """Seal a list of record dicts into the chain and persist them."""
        if not record_dicts:
            return
        conn = self._connect()
        try:
            with self._lock:
                prev = self._last_record_hash(conn)
                rows = []
                for rec in record_dicts:
                    content_hash = rec.get("content_hash") or integrity.compute_content_hash(rec)
                    sealed_at = datetime.utcnow().isoformat() + "Z"
                    record_hash = integrity.compute_record_hash(prev, content_hash, sealed_at)
                    seal = {
                        "algorithm": integrity.ALGORITHM,
                        "sealed_at": sealed_at,
                        "content_hash": content_hash,
                        "prev_hash": prev,
                        "record_hash": record_hash,
                    }
                    sealed_record = dict(rec)
                    sealed_record["content_hash"] = content_hash
                    sealed_record["seal"] = seal
                    rows.append((
                        rec.get("violation_id", ""),
                        rec.get("timestamp", ""),
                        rec.get("violation_type", ""),
                        float(rec.get("confidence", 0.0) or 0.0),
                        rec.get("severity", ""),
                        rec.get("vehicle_plate", "") or "",
                        content_hash, prev, record_hash, sealed_at,
                        json.dumps(sealed_record, cls=NumpyEncoder),
                    ))
                    prev = record_hash
                conn.executemany(
                    """INSERT INTO violations
                       (violation_id, timestamp, violation_type, confidence, severity,
                        vehicle_plate, content_hash, prev_hash, record_hash, sealed_at, record_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    rows,
                )
                conn.commit()
        finally:
            conn.close()

    # ── public interface (preserved) ──
    def add_record(self, package: EvidencePackage):
        self._insert_records([package.to_dict()])
        self._load()

    def add_records(self, packages: list):
        self._insert_records([p.to_dict() for p in packages])
        self._load()

    def search(self, violation_type: str = None, plate: str = None,
               date_from: str = None, date_to: str = None,
               min_confidence: float = 0.0) -> list:
        """Search records with filters."""
        results = self.records

        if violation_type:
            results = [r for r in results if r["violation_type"] == violation_type]
        if plate:
            results = [r for r in results if plate.upper() in r.get("vehicle_plate", "").upper()]
        if date_from:
            results = [r for r in results if r["timestamp"] >= date_from]
        if date_to:
            results = [r for r in results if r["timestamp"] <= date_to]
        if min_confidence > 0:
            results = [r for r in results if r["confidence"] >= min_confidence]

        return results

    def get_dataframe(self) -> pd.DataFrame:
        """Get records as a pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=["violation_id", "timestamp", "violation_type",
                                         "confidence", "severity", "vehicle_plate"])
        return pd.DataFrame(self.records)

    def _load(self, load_images: bool = False):
        """Refresh the in-memory mirror from database (oldest -> newest)."""
        conn = self._connect()
        try:
            with self._lock:
                if self.is_postgres and not load_images:
                    query = "SELECT CAST(record_json AS jsonb) - 'evidence_image' - 'plate_crop_image' AS record_json FROM violations ORDER BY id ASC"
                elif not self.is_postgres and not load_images:
                    try:
                        query = "SELECT json_remove(record_json, '$.evidence_image', '$.plate_crop_image') AS record_json FROM violations ORDER BY id ASC"
                        cur = conn.execute(query)
                    except (sqlite3.OperationalError, AttributeError):
                        query = "SELECT record_json FROM violations ORDER BY id ASC"
                else:
                    query = "SELECT record_json FROM violations ORDER BY id ASC"
                
                cur = conn.execute(query)
                records = []
                for row in cur.fetchall():
                    val = row["record_json"]
                    if isinstance(val, dict):
                        records.append(val)
                    elif isinstance(val, str):
                        records.append(json.loads(val))
                    else:
                        records.append(val)
                self.records = records
        finally:
            conn.close()

    # ── tamper-evidence API ──
    def get_seals(self) -> list:
        """Ordered list of integrity seals (for chain verification)."""
        return [r.get("seal") for r in self.records if r.get("seal")]

    def verify_chain(self) -> dict:
        """Verify the append-only hash chain across all stored records."""
        return integrity.verify_chain(self.get_seals())

    def _prune_oldest(self, n: int):
        """Delete the N oldest records to cap storage."""
        if n <= 0:
            return
        conn = self._connect()
        try:
            with self._lock:
                conn.execute(f"DELETE FROM violations WHERE id IN (SELECT id FROM violations ORDER BY id ASC LIMIT {int(n)})")
                conn.commit()
        finally:
            conn.close()
        self._load()



class AnalyticsEngine:
    """Generate statistics and reports from violation records."""

    def __init__(self, database: ViolationDatabase):
        self.db = database

    def summary_stats(self) -> dict:
        """Overall statistics."""
        df = self.db.get_dataframe()
        if df.empty:
            return {"total_violations": 0}

        return {
            "total_violations": len(df),
            "violation_types": dict(Counter(df["violation_type"])),
            "severity_distribution": dict(Counter(df["severity"])),
            "avg_confidence": round(df["confidence"].mean(), 3),
            "plates_identified": int((df["vehicle_plate"] != "").sum()),
            "high_confidence_count": int((df["confidence"] >= 0.7).sum()),
        }

    def violation_type_breakdown(self) -> dict:
        """Per-violation-type stats."""
        df = self.db.get_dataframe()
        if df.empty:
            return {}

        breakdown = {}
        for vtype in df["violation_type"].unique():
            subset = df[df["violation_type"] == vtype]
            breakdown[vtype] = {
                "count": len(subset),
                "avg_confidence": round(subset["confidence"].mean(), 3),
                "high_severity_count": int((subset["severity"] == "high").sum()),
            }
        return breakdown

    def time_trend(self) -> list:
        """Violations over time (grouped by date)."""
        df = self.db.get_dataframe()
        if df.empty:
            return []

        df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
        trend = df.groupby("date").size().reset_index(name="count")
        return trend.to_dict("records")

    def top_offenders(self, n: int = 10) -> list:
        """Most frequently caught plates."""
        df = self.db.get_dataframe()
        if df.empty:
            return []

        plates = df[df["vehicle_plate"] != ""]["vehicle_plate"]
        top = plates.value_counts().head(n)
        return [{"plate": plate, "violations": int(count)} for plate, count in top.items()]

    def generate_report(self) -> dict:
        """Full analytics report."""
        return {
            "generated_at": datetime.now().isoformat(),
            "summary": self.summary_stats(),
            "breakdown": self.violation_type_breakdown(),
            "trends": self.time_trend(),
            "top_offenders": self.top_offenders(),
        }
