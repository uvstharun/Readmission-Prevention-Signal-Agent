"""
Main Orchestrator Agent: Coordinates all sub-agents for a complete discharge processing pipeline.
"""
from typing import Dict, Optional
from datetime import datetime
from src.agents.risk_scoring_agent import RiskScoringAgent
from src.agents.patient_context_agent import PatientContextAgent
from src.agents.care_gap_agent import CareGapAgent
from src.agents.workflow_trigger_agent import WorkflowTriggerAgent
from src.llm.clinical_narrator import ClinicalNarrator
from src.utils.database import save_patient, save_risk_score, initialize_database
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReadmissionPreventionOrchestrator:
    """
    Orchestrates the full readmission prevention pipeline for a newly discharged patient.

    Pipeline:
    1. PatientContextAgent - builds structured patient context
    2. RiskScoringAgent - scores patient for 30-day readmission risk
    3. CareGapAgent - identifies missing care transition interventions
    4. ClinicalNarrator - generates LLM clinical narrative
    5. WorkflowTriggerAgent - triggers appropriate interventions and monitoring
    """

    def __init__(self):
        initialize_database()
        self.context_agent = PatientContextAgent()
        self.risk_agent = RiskScoringAgent()
        self.care_gap_agent = CareGapAgent()
        self.narrator = ClinicalNarrator()
        self.workflow_agent = WorkflowTriggerAgent()

    def process_discharge(self, patient_data: Dict, use_llm: bool = True) -> Dict:
        """
        Process a new discharge event through the full agent pipeline.

        Args:
            patient_data: Complete patient discharge record
            use_llm: Whether to generate LLM clinical narrative (requires API key)

        Returns:
            Complete patient action plan with risk score, narrative, care gaps, and triggered interventions
        """
        patient_id = patient_data.get("patient_id", "UNKNOWN")
        start_time = datetime.now()
        logger.info(f"=" * 60)
        logger.info(f"Processing discharge: {patient_id}")
        logger.info(f"=" * 60)

        pipeline_result = {
            "patient_id": patient_id,
            "pipeline_start": start_time.isoformat(),
            "steps_completed": [],
            "errors": [],
        }

        try:
            # Step 1: Save patient to database
            save_patient(patient_data)
            logger.info(f"[1/5] Saved patient record for {patient_id}")
            pipeline_result["steps_completed"].append("save_patient")

            # Step 2: Build patient context
            patient_context = self.context_agent.run(patient_data)
            pipeline_result["patient_context"] = patient_context
            pipeline_result["steps_completed"].append("patient_context")
            logger.info(f"[2/5] Patient context built | flags={len(patient_context.get('risk_flags', []))}")

            # Step 3: Risk scoring
            risk_result = self.risk_agent.run(patient_data)
            pipeline_result["risk_result"] = risk_result
            pipeline_result["steps_completed"].append("risk_scoring")
            logger.info(f"[3/5] Risk scored | score={risk_result['risk_score']} tier={risk_result['risk_tier']}")

            # Step 4: Care gap analysis
            care_gap_result = self.care_gap_agent.run(patient_data, risk_result)
            pipeline_result["care_gap_result"] = care_gap_result
            pipeline_result["steps_completed"].append("care_gap_analysis")
            logger.info(f"[4/5] Care gaps: {care_gap_result['total_gaps']} identified")

            # Step 5: Clinical narrative (LLM)
            clinical_narrative = {}
            if use_llm:
                try:
                    clinical_narrative = self.narrator.generate_narrative(
                        patient_data=patient_data,
                        risk_result=risk_result,
                        care_gaps=care_gap_result.get("care_gaps", []),
                    )
                    pipeline_result["steps_completed"].append("clinical_narrative")
                    logger.info(f"[5a/5] Clinical narrative generated via LLM")
                except Exception as e:
                    logger.warning(f"LLM narrative failed, using fallback: {e}")
                    clinical_narrative = self.narrator._fallback_narrative(patient_data, risk_result)
                    pipeline_result["errors"].append(f"LLM narrative fallback: {str(e)}")
            else:
                clinical_narrative = self.narrator._fallback_narrative(patient_data, risk_result)

            pipeline_result["clinical_narrative"] = clinical_narrative

            # Save risk score to DB
            save_risk_score(
                patient_id=patient_id,
                risk_score=risk_result["risk_score"],
                risk_tier=risk_result["risk_tier"],
                top_features=risk_result.get("top_risk_drivers", []),
                clinical_narrative=clinical_narrative,
            )

            # Step 6: Trigger workflows
            workflow_result = self.workflow_agent.run(
                patient_data=patient_data,
                risk_result=risk_result,
                care_gap_result=care_gap_result,
                patient_context=patient_context,
                clinical_narrative=clinical_narrative,
            )
            pipeline_result["workflow_result"] = workflow_result
            pipeline_result["steps_completed"].append("workflow_trigger")
            logger.info(f"[5/5] Workflows triggered | interventions={workflow_result['interventions_triggered']}")

        except Exception as e:
            logger.error(f"Pipeline error for {patient_id}: {e}", exc_info=True)
            pipeline_result["errors"].append(str(e))

        # Final summary
        elapsed = (datetime.now() - start_time).total_seconds()
        pipeline_result["pipeline_end"] = datetime.now().isoformat()
        pipeline_result["elapsed_seconds"] = round(elapsed, 2)
        pipeline_result["success"] = len(pipeline_result["errors"]) == 0

        tier = pipeline_result.get("risk_result", {}).get("risk_tier", "UNKNOWN")
        score = pipeline_result.get("risk_result", {}).get("risk_score", 0)
        logger.info(f"Pipeline complete for {patient_id}: tier={tier} score={score} time={elapsed:.1f}s")

        return pipeline_result
