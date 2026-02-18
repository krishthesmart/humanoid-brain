"""Organizing policy."""

from typing import Any, Dict, List

from .base_policy import TaskPolicy


class OrganizingPolicy(TaskPolicy):
    def __init__(self):
        super().__init__(task_name="organizing")

    def plan(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        env_state = observation.get("env_state", {}) or {}
        return [
            {"type": "grasp", "target_frame": "misplaced_item", "pose": None, "params": {}},
            {"type": "move", "target_frame": "storage_area", "pose": env_state.get("storage_pose"), "params": {}},
            {"type": "place", "target_frame": "storage_bin", "pose": None, "params": {"order": "category"}},
        ]
