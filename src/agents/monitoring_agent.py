"""
Monitoring Agent: Maintains 30-day watchlist, detects escalation signals, closes cases.
"""
import random
from typing import Dict, List
from datetime import datetime, timedelta
from src.utils.logger import get_logger, log_agent_action
from src.utils.database import get_active_watchlist, save_agent_decision, get_connection
from src.utils.config import config

logger = get_logger(__name__)

# Simulated monitoring signal types
MONITORING_SIGNALS = [
    {"signal": "missed_followup_appointment", "severity": "HIGH", "probability": 0.15},
    {"signal": "ed_visit_detected", "severity": "HIGH", "probability": 0.08},
    {"signal": "medication_non_adherence", "severity": "MODERATE", "probability": 0.12},
    {"signal": "vitals_deterioration", "severity": "HIGH", "probability": 0.05},
    {"signal": "patient_reported_worsening", "severity": "MODERATE", "probability": 0.10},
    {"signal": "missed_lab_followup", "severity": "LOW", "probability": 0.18},
]


class MonitoringAgent:
    """
    Monitors post-discharge patients, detects escalation signals,
    and closes cases at 30 days.
    """

    def run_monitoring_cycle(self) -> Dict:
        """
        Run one monitoring cycle across all active watchlist patients.
        Simulates signal detection and escalates where appropriate.
        """
        watchlist = get_active_watchlist()
        logger.info(f"Monitoring cycle: {len(watchlist)} active patients")

        processed = 0
        escalated = 0
        closed = 0

        for patient in watchlist:
            patient_id = patient["patient_id"]
            signals = self._detect_signals(patient)

            if signals:
                should_escalate = any(s["severity"] == "HIGH" for s in signals)
                self._update_watchlist_status(
                    patient_id,
                    signals,
                    "ESCALATED" if should_escalate else "SIGNAL_DETECTED",
                )
                if should_escalate:
                    escalated += 1
                    logger.warning(f"ESCALATION: Patient {patient_id} | signals={[s['signal'] for s in signals]}")

            # Check if monitoring window is closed
            if self._is_monitoring_complete(patient):
                self._close_case(patient_id, patient)
                closed += 1
            else:
                self._update_days_remaining(patient_id, patient)

            processed += 1

        result = {
            "cycle_timestamp": datetime.now().isoformat(),
            "patients_monitored": processed,
            "patients_escalated": escalated,
            "cases_closed": closed,
        }

        logger.info(f"Monitoring cycle complete: {result}")
        return result

    def _detect_signals(self, patient: Dict) -> List[Dict]:
        """Simulate monitoring signal detection for a patient."""
        detected = []
        risk_tier = patient.get("risk_tier", "LOW")
        # Higher risk patients have higher signal probability
        multiplier = {"HIGH": 1.5, "MODERATE": 1.0, "LOW": 0.5}.get(risk_tier, 1.0)

        for signal_def in MONITORING_SIGNALS:
            prob = signal_def["probability"] * multiplier
            if random.random() < prob:
                detected.append({
                    "signal": signal_def["signal"],
                    "severity": signal_def["severity"],
                    "detected_at": datetime.now().isoformat(),
                })
        return detected

    def _is_monitoring_complete(self, patient: Dict) -> bool:
        """Check if 30-day monitoring window has passed."""
        end_date_str = patient.get("monitoring_end_date")
        if not end_date_str:
            return False
        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            return datetime.now() >= end_date
        except Exception:
            return False

    def _update_watchlist_status(self, patient_id: str, signals: List[Dict], status: str):
        """Update patient watchlist status with detected signals."""
        import json
        try:
            with get_connection() as conn:
                conn.execute("""
                    UPDATE monitoring_watchlist
                    SET escalation_status = ?,
                        signals_detected = ?,
                        last_checked_at = datetime('now')
                    WHERE patient_id = ?
                """, (status, json.dumps(signals), patient_id))
        except Exception as e:
            logger.error(f"Failed to update watchlist status: {e}")

    def _update_days_remaining(self, patient_id: str, patient: Dict):
        """Update days remaining for a patient."""
        try:
            end_date_str = patient.get("monitoring_end_date", "")
            if end_date_str:
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                days_remaining = max(0, (end_date - datetime.now()).days)
                with get_connection() as conn:
                    conn.execute("""
                        UPDATE monitoring_watchlist
                        SET days_remaining = ?, last_checked_at = datetime('now')
                        WHERE patient_id = ?
                    """, (days_remaining, patient_id))
        except Exception as e:
            logger.error(f"Failed to update days remaining: {e}")

    def _close_case(self, patient_id: str, patient: Dict):
        """Close a monitoring case and record final outcome."""
        # Simulate actual readmission outcome
        original_tier = patient.get("risk_tier", "LOW")
        readmit_probs = {"HIGH": 0.28, "MODERATE": 0.15, "LOW": 0.07}
        actual_readmission = 1 if random.random() < readmit_probs.get(original_tier, 0.10) else 0

        try:
            with get_connection() as conn:
                conn.execute("""
                    INSERT INTO readmission_outcomes
                    (patient_id, original_risk_tier, original_risk_score, actual_readmission)
                    VALUES (?, ?, ?, ?)
                """, (patient_id, original_tier, patient.get("risk_score"), actual_readmission))

                conn.execute("""
                    UPDATE monitoring_watchlist SET days_remaining = 0 WHERE patient_id = ?
                """, (patient_id,))
        except Exception as e:
            logger.error(f"Failed to close case for {patient_id}: {e}")

        logger.info(f"Closed monitoring case: {patient_id} | readmitted={actual_readmission}")

        save_agent_decision(
            patient_id=patient_id,
            agent_name="MonitoringAgent",
            action="close_case",
            reasoning=f"30-day monitoring window complete for {original_tier} risk patient",
            output={"actual_readmission": actual_readmission},
        )
