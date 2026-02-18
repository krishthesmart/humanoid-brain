"""SDK API for task inference + policy planning."""

from __future__ import annotations

from typing import Any, Dict, Optional

from humanoid_brain.models.task_classifier import TaskClassifier
from humanoid_brain.policies.cleaning_policy import CleaningPolicy
from humanoid_brain.policies.cooking_policy import CookingPolicy
from humanoid_brain.policies.dishwashing_policy import DishwashingPolicy
from humanoid_brain.policies.laundry_policy import LaundryPolicy
from humanoid_brain.policies.organizing_policy import OrganizingPolicy
from humanoid_brain.telemetry.events import ErrorEvent, PolicyPlanEvent
from humanoid_brain.telemetry.logger import TelemetryLogger


class HumanoidBrain:
    """Main orchestrator for task classification and symbolic planning."""

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        min_confidence: float = 0.6,
        telemetry_logger: Optional[TelemetryLogger] = None,
    ):
        self.telemetry = telemetry_logger
        self.classifier = TaskClassifier(
            weights_path=weights_path,
            device=device,
            min_confidence=min_confidence,
            telemetry_logger=telemetry_logger,
        )
        self.policies = {
            "cleaning": CleaningPolicy(),
            "cooking": CookingPolicy(),
            "dishwashing": DishwashingPolicy(),
            "laundry": LaundryPolicy(),
            "organizing": OrganizingPolicy(),
        }

    def decide(self, image: Any, robot_state: Optional[Dict[str, Any]] = None, env_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run task prediction and produce symbolic sub-goals.

        Returns:
          {
            "task": str,
            "probs": Dict[str, float],
            "sub_goals": List[Dict[str, Any]],
            "unknown": bool
          }
        """
        try:
            pred = self.classifier.predict(image)
            label = pred["label"]
            probs = pred["probs"]

            if label == "unknown":
                result = {"task": "unknown", "probs": probs, "sub_goals": [], "unknown": True}
                if self.telemetry:
                    self.telemetry.log_event(PolicyPlanEvent(task="unknown", sub_goal_count=0, metadata={"reason": "low_confidence"}))
                return result

            policy = self.policies.get(label)
            if policy is None:
                result = {"task": label, "probs": probs, "sub_goals": [], "unknown": True}
                if self.telemetry:
                    self.telemetry.log_event(
                        ErrorEvent(source="HumanoidBrain.decide", message=f"No policy for label: {label}")
                    )
                return result

            observation = {"image": image, "robot_state": robot_state or {}, "env_state": env_state or {}}
            sub_goals = policy.plan(observation)
            if self.telemetry:
                self.telemetry.log_event(PolicyPlanEvent(task=label, sub_goal_count=len(sub_goals)))

            return {"task": label, "probs": probs, "sub_goals": sub_goals, "unknown": False}
        except Exception as exc:
            if self.telemetry:
                self.telemetry.log_event(
                    ErrorEvent(source="HumanoidBrain.decide", message=str(exc), details={"type": type(exc).__name__})
                )
            raise


def load_brain(weights_path: str, device: str = "cpu", min_confidence: float = 0.6, telemetry_logger: Optional[TelemetryLogger] = None) -> HumanoidBrain:
    """Factory to create HumanoidBrain."""
    return HumanoidBrain(
        weights_path=weights_path,
        device=device,
        min_confidence=min_confidence,
        telemetry_logger=telemetry_logger,
    )
