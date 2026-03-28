"""
Post-discharge monitoring scheduler and coordinator.
"""
from datetime import datetime
from src.agents.monitoring_agent import MonitoringAgent
from src.utils.logger import get_logger
from src.utils.database import get_active_watchlist

logger = get_logger(__name__)


class PostDischargeMonitor:
    """Coordinates post-discharge monitoring cycles."""

    def __init__(self):
        self.monitoring_agent = MonitoringAgent()

    def run_cycle(self) -> dict:
        """Run a full monitoring cycle."""
        logger.info(f"Post-discharge monitoring cycle starting at {datetime.now().isoformat()}")
        result = self.monitoring_agent.run_monitoring_cycle()
        return result

    def get_watchlist_summary(self) -> dict:
        """Get current watchlist status summary."""
        watchlist = get_active_watchlist()
        return {
            "total_active": len(watchlist),
            "high_risk": sum(1 for p in watchlist if p.get("risk_tier") == "HIGH"),
            "moderate_risk": sum(1 for p in watchlist if p.get("risk_tier") == "MODERATE"),
            "escalated": sum(1 for p in watchlist if p.get("escalation_status") == "ESCALATED"),
            "patients": watchlist,
        }
