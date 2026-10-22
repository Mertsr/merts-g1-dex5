"""The model-based object tracker on the head depth camera.

    ros2 launch g1_perception object_tracker.launch.py
    ros2 launch g1_perception object_tracker.launch.py config:=/path/to/my_line.yaml
    ros2 launch g1_perception object_tracker.launch.py rviz:=true   # watch what it does

Publishes /g1_object_tracker/detections (vision_msgs/Detection3DArray, camera frame), which
g1_object_pose_source turns into /objects with object_source:=depth_tracker. Needs depth +
camera_info and TF from odom to the camera: in the simulator, sensors:=true.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("g1_perception")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=os.path.join(share, "config", "object_tracker_conveyor.yaml"),
                description="The objects, their models and priors. See g1_perception/README.md.",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Open RViz on the tracker's debug views (rviz/perception.rviz).",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="perception_rviz",
                arguments=["-d", os.path.join(share, "rviz", "perception.rviz")],
                output="log",
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
            Node(
                package="g1_perception",
                executable="g1_object_tracker",
                name="g1_object_tracker",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
        ]
    )
