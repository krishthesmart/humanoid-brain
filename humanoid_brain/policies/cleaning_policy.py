"""Cleaning policy."""

from typing import Any, Dict, List

from .base_policy import TaskPolicy


class CleaningPolicy(TaskPolicy):
    def __init__(self):
        super().__init__(task_name="cleaning")

    def plan(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        env_state = observation.get("env_state", {}) or {}
        return [
            {"type": "move", "target_frame": "dirty_surface", "pose": env_state.get("dirty_surface_pose"), "params": {}},
            {"type": "wipe", "target_frame": "dirty_surface", "pose": None, "params": {"passes": 3}},
            {"type": "place", "target_frame": "cleaning_tool_station", "pose": env_state.get("tool_station_pose"), "params": {}},
        ]
