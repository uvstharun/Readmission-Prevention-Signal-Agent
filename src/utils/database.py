"""
SQLite database management for the readmission prevention agent.
Creates and manages all tables for patient records, risk scores, interventions, and monitoring.
"""
import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
from contextlib import contextmanager
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = config.DATABASE_URL.replace("sqlite:///", "")


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database():
    """Create all database tables if they don't exist."""
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                age INTEGER,
                gender TEXT,
                race TEXT,
                insurance_type TEXT,
                primary_diagnosis_code TEXT,
                discharge_disposition TEXT,
                discharge_date TEXT,
                length_of_stay_days INTEGER,
                charlson_comorbidity_index INTEGER,
                attending_department TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_tier TEXT NOT NULL,
                top_features TEXT,
                clinical_narrative TEXT,
                scored_at TEXT DEFAULT (datetime('now')),
                model_version TEXT DEFAULT 'v1.0'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                action TEXT NOT NULL,
                reasoning TEXT,
                output TEXT,
                decided_at TEXT DEFAULT (datetime('now'))
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS triggered_interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                intervention_name TEXT NOT NULL,
                intervention_type TEXT,
                responsible_team TEXT,
                priority TEXT,
                status TEXT DEFAULT 'PENDING',
                triggered_at TEXT DEFAULT (datetime('now')),
                completed_at TEXT,
                outcome TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS monitoring_watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL UNIQUE,
                risk_tier TEXT NOT NULL,
                risk_score REAL,
                discharge_date TEXT,
                monitoring_end_date TEXT,
                days_remaining INTEGER,
                escalation_status TEXT DEFAULT 'STABLE',
                signals_detected TEXT,
                added_at TEXT DEFAULT (datetime('now')),
                last_checked_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS readmission_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                original_risk_tier TEXT,
                original_risk_score REAL,
                actual_readmission INTEGER,
                readmission_date TEXT,
                readmission_diagnosis TEXT,
                case_closed_at TEXT DEFAULT (datetime('now'))
            )
        """)

    logger.info("Database initialized successfully")


def save_patient(patient_data: Dict):
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO patients
            (patient_id, age, gender, race, insurance_type, primary_diagnosis_code,
             discharge_disposition, discharge_date, length_of_stay_days,
             charlson_comorbidity_index, attending_department)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_data.get("patient_id"),
            patient_data.get("age"),
            patient_data.get("gender"),
            patient_data.get("race"),
            patient_data.get("insurance_type"),
            patient_data.get("primary_diagnosis_code"),
            patient_data.get("discharge_disposition"),
            patient_data.get("discharge_date"),
            patient_data.get("length_of_stay_days"),
            patient_data.get("charlson_comorbidity_index"),
            patient_data.get("attending_department"),
        ))


def save_risk_score(patient_id: str, risk_score: float, risk_tier: str,
                    top_features: list, clinical_narrative: dict):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO risk_scores (patient_id, risk_score, risk_tier, top_features, clinical_narrative)
            VALUES (?, ?, ?, ?, ?)
        """, (
            patient_id,
            risk_score,
            risk_tier,
            json.dumps(top_features),
            json.dumps(clinical_narrative),
        ))


def save_agent_decision(patient_id: str, agent_name: str, action: str,
                        reasoning: str, output: dict):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO agent_decisions (patient_id, agent_name, action, reasoning, output)
            VALUES (?, ?, ?, ?, ?)
        """, (patient_id, agent_name, action, reasoning, json.dumps(output)))


def save_intervention(patient_id: str, intervention: Dict):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO triggered_interventions
            (patient_id, intervention_name, intervention_type, responsible_team, priority)
            VALUES (?, ?, ?, ?, ?)
        """, (
            patient_id,
            intervention.get("name"),
            intervention.get("type"),
            intervention.get("responsible_team"),
            intervention.get("priority"),
        ))


def add_to_watchlist(patient_id: str, risk_tier: str, risk_score: float,
                     discharge_date: str, monitoring_days: int = 30):
    try:
        discharge_dt = datetime.strptime(discharge_date, "%Y-%m-%d")
    except Exception:
        discharge_dt = datetime.now()
    end_date = discharge_dt + timedelta(days=monitoring_days)

    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO monitoring_watchlist
            (patient_id, risk_tier, risk_score, discharge_date, monitoring_end_date, days_remaining)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            patient_id, risk_tier, risk_score,
            discharge_date, end_date.strftime("%Y-%m-%d"), monitoring_days
        ))


def get_active_watchlist() -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM monitoring_watchlist
            WHERE days_remaining > 0
            ORDER BY risk_score DESC
        """).fetchall()
        return [dict(r) for r in rows]


def get_patient_risk_history(patient_id: str) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM risk_scores WHERE patient_id = ? ORDER BY scored_at DESC
        """, (patient_id,)).fetchall()
        return [dict(r) for r in rows]


def get_patient_interventions(patient_id: str) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM triggered_interventions WHERE patient_id = ? ORDER BY triggered_at DESC
        """, (patient_id,)).fetchall()
        return [dict(r) for r in rows]


def get_dashboard_summary() -> Dict:
    today = datetime.now().strftime("%Y-%m-%d")
    with get_connection() as conn:
        total_scored_today = conn.execute(
            "SELECT COUNT(*) FROM risk_scores WHERE DATE(scored_at) = ?", (today,)
        ).fetchone()[0]

        high_risk_count = conn.execute(
            "SELECT COUNT(*) FROM risk_scores WHERE risk_tier = 'HIGH' AND DATE(scored_at) = ?", (today,)
        ).fetchone()[0]

        interventions_triggered = conn.execute(
            "SELECT COUNT(*) FROM triggered_interventions WHERE DATE(triggered_at) = ?", (today,)
        ).fetchone()[0]

        readmission_rate = conn.execute(
            "SELECT AVG(actual_readmission) FROM readmission_outcomes"
        ).fetchone()[0]

    return {
        "total_patients_scored_today": total_scored_today,
        "high_risk_count_today": high_risk_count,
        "interventions_triggered_today": interventions_triggered,
        "overall_readmission_rate": float(readmission_rate or 0),
        "active_watchlist_count": len(get_active_watchlist()),
    }


try:
    initialize_database()
except Exception:
    pass
