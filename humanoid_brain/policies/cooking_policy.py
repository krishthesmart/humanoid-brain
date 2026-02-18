"""Cooking policy."""

from typing import Any, Dict, List

from .base_policy import TaskPolicy


class CookingPolicy(TaskPolicy):
    def __init__(self):
        super().__init__(task_name="cooking")

    def plan(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        env_state = observation.get("env_state", {}) or {}
        return [
            {"type": "move", "target_frame": "prep_counter", "pose": env_state.get("prep_counter_pose"), "params": {}},
            {"type": "grasp", "target_frame": "ingredient", "pose": None, "params": {"tool": "gripper"}},
            {"type": "place", "target_frame": "cutting_board", "pose": env_state.get("cutting_board_pose"), "params": {}},
        ]
