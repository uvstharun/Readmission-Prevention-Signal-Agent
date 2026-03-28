"""
Workflow Trigger Agent: Takes risk tier + care gaps, triggers appropriate interventions,
generates case manager alert.
"""
from typing import Dict, List
from src.workflows.risk_stratification import get_intervention_bundle
from src.workflows.alert_engine import format_case_manager_alert
from src.utils.logger import get_logger, log_agent_action
from src.utils.database import save_agent_decision, save_intervention, add_to_watchlist
from src.utils.config import config

logger = get_logger(__name__)


class WorkflowTriggerAgent:
    """Triggers appropriate care transition workflows based on risk tier and care gaps."""

    def run(
        self,
        patient_data: Dict,
        risk_result: Dict,
        care_gap_result: Dict,
        patient_context: Dict,
        clinical_narrative: Dict,
    ) -> Dict:
        patient_id = patient_data.get("patient_id", "UNKNOWN")
        risk_tier = risk_result.get("risk_tier", "LOW")

        logger.info(f"Triggering workflows for patient {patient_id} | tier={risk_tier}")

        # Get intervention bundle for this risk tier
        intervention_bundle = get_intervention_bundle(risk_tier, care_gap_result.get("care_gaps", []))

        # Save each triggered intervention to DB
        triggered = []
        for intervention in intervention_bundle:
            try:
                save_intervention(patient_id, intervention)
                triggered.append(intervention)
                logger.debug(f"Triggered intervention: {intervention.get('name')} for {patient_id}")
            except Exception as e:
                logger.error(f"Failed to save intervention: {e}")

        # Add to monitoring watchlist if high or moderate risk
        watchlist_added = False
        if risk_tier in ["HIGH", "MODERATE"]:
            try:
                add_to_watchlist(
                    patient_id=patient_id,
                    risk_tier=risk_tier,
                    risk_score=risk_result.get("risk_score", 0),
                    discharge_date=patient_data.get("discharge_date", "2024-01-01"),
                    monitoring_days=config.MONITORING_WINDOW_DAYS,
                )
                watchlist_added = True
                logger.info(f"Added {patient_id} to 30-day monitoring watchlist")
            except Exception as e:
                logger.error(f"Failed to add to watchlist: {e}")

        # Format case manager alert
        alert = format_case_manager_alert(
            patient_data=patient_data,
            risk_result=risk_result,
            clinical_narrative=clinical_narrative,
            care_gaps=care_gap_result.get("care_gaps", []),
            interventions=triggered,
        )

        result = {
            "patient_id": patient_id,
            "risk_tier": risk_tier,
            "interventions_triggered": len(triggered),
            "intervention_list": triggered,
            "watchlist_added": watchlist_added,
            "case_manager_alert": alert,
        }

        log_agent_action("WorkflowTriggerAgent", patient_id, "trigger_workflows",
                         f"interventions={len(triggered)} watchlist={watchlist_added}")

        save_agent_decision(
            patient_id=patient_id,
            agent_name="WorkflowTriggerAgent",
            action="trigger_workflows",
            reasoning=f"Triggered {len(triggered)} interventions for {risk_tier} risk patient. Watchlist: {watchlist_added}",
            output=result,
        )

        return result
