"""Example ROS2 launch file for task_brain_node.

Expected topics:
- Subscribed:
  - /camera/color/image_raw (sensor_msgs/Image)
  - /robot/state (std_msgs/String JSON)
- Published:
  - /task_brain/decision (std_msgs/String JSON)
  - /task_brain/sub_goals (std_msgs/String JSON)
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="humanoid_brain",
                executable="task_brain_node",
                name="task_brain_node",
                output="screen",
                parameters=[
                    {
                        "weights_path": "best_licensed_balanced.pt",
                        "device": "cpu",
                    }
                ],
            )
        ]
    )
