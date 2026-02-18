"""Base task policy interface."""

from typing import Any, Dict, List


class TaskPolicy:
    """Abstract symbolic task policy."""

    def __init__(self, task_name: str):
        self.task_name = task_name

    def plan(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate symbolic sub-goals for downstream robot control stack.

        observation:
          - image: camera frame
          - robot_state: dict with robot pose/state
          - env_state: dict with environment metadata
        """
        raise NotImplementedError
