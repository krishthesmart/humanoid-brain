"""Telemetry event type definitions."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BaseEvent:
    """Base telemetry event."""

    event_type: str
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskDecisionEvent(BaseEvent):
    """Event for model task prediction outputs."""

    label: str = "unknown"
    probs: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def __init__(self, label: str, probs: Dict[str, float], confidence: float):
        super().__init__(event_type="task_decision")
        self.label = label
        self.probs = probs
        self.confidence = confidence


@dataclass
class PolicyPlanEvent(BaseEvent):
    """Event for policy planning outputs."""

    task: str = "unknown"
    sub_goal_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, task: str, sub_goal_count: int, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(event_type="policy_plan")
        self.task = task
        self.sub_goal_count = sub_goal_count
        self.metadata = metadata


@dataclass
class ErrorEvent(BaseEvent):
    """Event for runtime errors."""

    source: str = "unknown"
    message: str = ""
    details: Optional[Dict[str, Any]] = None

    def __init__(self, source: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(event_type="error")
        self.source = source
        self.message = message
        self.details = details
