"""
SQLite Temporal Persistence Store for Longitudinal Banana Severity Tracking (FR9).

Stores sequential observation histories per plant_id:
- Plant, zone, leaf, date identifiers
- Diagnosed disease type and visual foliar metrics (scope %, impact score, stover grade)
- Microclimate urgency score and fused composite severity score
- Fused severity category and fusion methodology
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


DEFAULT_DB_PATH = "data/severity_history.db"


class SeverityDatabase:
    """
    Manages longitudinal severity observation storage and retrieval using SQLite.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Creates severity_records table and indexes if not existing."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS severity_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plant_id TEXT NOT NULL,
                    zone_id TEXT NOT NULL,
                    leaf_id TEXT NOT NULL,
                    observation_date TEXT NOT NULL,
                    disease_label TEXT NOT NULL,
                    scope_pct REAL NOT NULL,
                    impact_score REAL NOT NULL,
                    stover_grade INTEGER,
                    urgency_score REAL NOT NULL,
                    severity_score REAL NOT NULL,
                    severity_category TEXT NOT NULL,
                    fusion_method TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_plant_date
                ON severity_records (plant_id, observation_date ASC);
            """)
            conn.commit()

    def insert_record(
        self,
        plant_id: str,
        zone_id: str,
        leaf_id: str,
        observation_date: str,
        disease_label: str,
        scope_pct: float,
        impact_score: float,
        urgency_score: float,
        severity_score: float,
        severity_category: str,
        fusion_method: str,
        stover_grade: Optional[int] = None
    ) -> int:
        """
        Inserts a new observation record into the database.

        Returns:
            The generated integer record ID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO severity_records (
                    plant_id, zone_id, leaf_id, observation_date,
                    disease_label, scope_pct, impact_score, stover_grade,
                    urgency_score, severity_score, severity_category, fusion_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plant_id, zone_id, leaf_id, observation_date,
                disease_label, float(scope_pct), float(impact_score),
                stover_grade if stover_grade is not None else None,
                float(urgency_score), float(severity_score),
                severity_category, fusion_method
            ))
            conn.commit()
            return cursor.lastrowid

    def get_plant_history(
        self,
        plant_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves chronological history for a specific plant_id, ordered by observation_date ASC.

        Args:
            plant_id: Plant identifier string.
            limit: Optional maximum number of recent records to return.

        Returns:
            List of dictionaries containing row data.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if limit is not None and limit > 0:
                cursor.execute("""
                    SELECT * FROM (
                        SELECT * FROM severity_records
                        WHERE plant_id = ?
                        ORDER BY observation_date DESC
                        LIMIT ?
                    ) ORDER BY observation_date ASC
                """, (plant_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM severity_records
                    WHERE plant_id = ?
                    ORDER BY observation_date ASC
                """, (plant_id,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def count_plant_observations(self, plant_id: str) -> int:
        """Returns the number of historical records recorded for this plant."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM severity_records WHERE plant_id = ?", (plant_id,))
            return cursor.fetchone()[0]

    def clear_records(self):
        """Truncates all records from the database (used for testing)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM severity_records;")
            conn.commit()
