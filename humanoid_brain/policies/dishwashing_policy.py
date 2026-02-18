"""Dishwashing task policy."""

from typing import Any, Dict, List

from .base_policy import TaskPolicy


class DishwashingPolicy(TaskPolicy):
    """Simple rule-based dishwashing sequence."""

    def __init__(self):
        super().__init__(task_name="dishwashing")

    def plan(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        env_state = observation.get("env_state", {}) or {}
        rack_pose = env_state.get("dishwasher_rack_pose")
        sink_pose = env_state.get("sink_pose")

        return [
            {
                "type": "move",
                "target_frame": "sink",
                "pose": sink_pose,
                "params": {"speed": "normal"},
            },
            {
                "type": "grasp",
                "target_frame": "plate",
                "pose": None,
                "params": {"grasp_mode": "top"},
            },
            {
                "type": "move",
                "target_frame": "dishwasher_rack",
                "pose": rack_pose,
                "params": {"speed": "slow"},
            },
            {
                "type": "place",
                "target_frame": "dishwasher_slot",
                "pose": None,
                "params": {"orientation": "upright"},
            },
        ]
