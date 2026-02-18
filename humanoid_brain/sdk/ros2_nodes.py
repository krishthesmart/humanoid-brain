"""ROS2 integration node for HumanoidBrain.

This module is ROS2-optional. Importing works without ROS2 installed; running node requires:
- rclpy
- sensor_msgs
- std_msgs
- cv_bridge
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from humanoid_brain.sdk.inference_api import HumanoidBrain
from humanoid_brain.telemetry.events import ErrorEvent
from humanoid_brain.telemetry.logger import TelemetryLogger

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from cv_bridge import CvBridge
except Exception:  # pragma: no cover - optional dependency
    rclpy = None
    Node = object  # type: ignore[assignment]
    Image = object  # type: ignore[assignment]
    String = object  # type: ignore[assignment]
    CvBridge = None


@dataclass
class TaskDecision:
    """
    Example custom message shape (suggested .msg):
      string task
      float32 confidence
      string[] class_names
      float32[] class_probs
    """

    task: str
    confidence: float
    probs: Dict[str, float]


@dataclass
class SubGoalArray:
    """
    Example custom message shape (suggested .msg):
      string task
      string json_sub_goals
    """

    task: str
    sub_goals: List[Dict[str, Any]]


class TaskBrainNode(Node):  # type: ignore[misc]
    """ROS2 node that consumes camera frames and publishes task decisions."""

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        image_topic: str = "/camera/color/image_raw",
        robot_state_topic: str = "/robot/state",
        decision_topic: str = "/task_brain/decision",
        sub_goals_topic: str = "/task_brain/sub_goals",
        telemetry_logger: Optional[TelemetryLogger] = None,
    ):
        super().__init__("task_brain_node")
        self.telemetry = telemetry_logger
        self.brain = HumanoidBrain(weights_path=weights_path, device=device, telemetry_logger=telemetry_logger)
        self.bridge = CvBridge() if CvBridge else None
        self.latest_robot_state: Dict[str, Any] = {}

        self.image_sub = self.create_subscription(Image, image_topic, self._on_image, 10)
        self.robot_state_sub = self.create_subscription(String, robot_state_topic, self._on_robot_state, 10)

        # TODO: replace std_msgs/String with real custom ROS2 messages.
        self.decision_pub = self.create_publisher(String, decision_topic, 10)
        self.sub_goals_pub = self.create_publisher(String, sub_goals_topic, 10)

    def _on_robot_state(self, msg: String) -> None:
        try:
            self.latest_robot_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.latest_robot_state = {}

    def _image_to_np(self, msg: Image) -> np.ndarray:
        if self.bridge is None:
            raise RuntimeError("cv_bridge is required for ROS image conversion")
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        return np.asarray(cv_img, dtype=np.uint8)

    def _on_image(self, msg: Image) -> None:
        try:
            image_np = self._image_to_np(msg)
            result = self.brain.decide(image_np, robot_state=self.latest_robot_state, env_state={})
            probs = result["probs"]
            task = result["task"]
            confidence = probs.get(task, 0.0) if task != "unknown" else max(probs.values(), default=0.0)

            decision_payload = {
                "task": task,
                "confidence": confidence,
                "probs": probs,
                "unknown": result["unknown"],
            }
            sub_goals_payload = {"task": task, "sub_goals": result["sub_goals"]}

            decision_msg = String()
            decision_msg.data = json.dumps(decision_payload)
            self.decision_pub.publish(decision_msg)

            goals_msg = String()
            goals_msg.data = json.dumps(sub_goals_payload)
            self.sub_goals_pub.publish(goals_msg)
        except Exception as exc:
            if self.telemetry:
                self.telemetry.log_event(
                    ErrorEvent(source="TaskBrainNode._on_image", message=str(exc), details={"type": type(exc).__name__})
                )
            self.get_logger().error(f"TaskBrainNode error: {exc}")


def main() -> None:
    """CLI entrypoint for the ROS2 node."""
    if rclpy is None:
        raise RuntimeError("ROS2 dependencies not available. Install rclpy + sensor_msgs + std_msgs + cv_bridge.")

    rclpy.init()
    node = TaskBrainNode(weights_path="best_licensed_balanced.pt")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
