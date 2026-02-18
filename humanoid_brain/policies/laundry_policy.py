"""Laundry policy."""

from typing import Any, Dict, List

from .base_policy import TaskPolicy


class LaundryPolicy(TaskPolicy):
    def __init__(self):
        super().__init__(task_name="laundry")

    def plan(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        env_state = observation.get("env_state", {}) or {}
        return [
            {"type": "move", "target_frame": "laundry_basket", "pose": env_state.get("basket_pose"), "params": {}},
            {"type": "grasp", "target_frame": "garment", "pose": None, "params": {"grasp_mode": "pinch"}},
            {"type": "place", "target_frame": "fold_table", "pose": env_state.get("fold_table_pose"), "params": {}},
        ]
