"""robot_state_publisher, ros2_control_node and the controller spawners.

No simulator, so this carries to hardware unchanged. G1LowCmdSystem owns all 29 body motors with
no onboard balance underneath: the policy spawned here is the balance controller.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

XACRO_PATH = os.path.join(
    get_package_share_directory("g1_description"), "urdf", "g1_lowcmd.urdf.xacro"
)
CONTROLLERS_YAML = os.path.join(
    get_package_share_directory("g1_controllers"), "config", "lowcmd_controllers.yaml"
)

# Forwards SIGTERM/INT to the `ros2 run` subprocess, which would otherwise be orphaned, and
# re-waits so launch sees a clean exit.
_SIGNAL_FORWARDING_WRAPPER = (
    "set -m; {command} & child=$!; "
    "trap 'kill -TERM -$child 2>/dev/null; wait $child' TERM INT; "
    "wait $child"
)


# Every body joint is claimed from the start: one the component sees unclaimed is left unpowered.
# The policy and the safety controller it chains into activate as one group, because chained
# reference interfaces only become claimable inside the switch that activates them.
def _controllers(pin_pelvis):
    """The spawn set, differing only in what drives the legs; the freeze and safety controllers
    claim the same 14 joints. The --inactive ones stay loaded for a later switch, such as
    activate_arm's or the safety controller's emergency."""
    legs = (
        [
            (["locomotion_freeze_controller"], []),
            (["locomotion_safety_controller", "agile_controller"], ["--inactive"]),
        ]
        if pin_pelvis
        else [
            (["locomotion_safety_controller", "agile_controller"], ["--activate-as-group"]),
            (["locomotion_freeze_controller"], ["--inactive"]),
        ]
    )
    return [
        (["joint_state_broadcaster"], []),
        (["imu_sensor_broadcaster"], []),
        (["waist_freeze_controller"], []),
        (["arm_freeze_controller"], []),
        *legs,
        (["arm_trajectory_controller"], ["--inactive"]),
        # Left Dex5 only: the right hand has no component, so a spawner for it would fail on
        # interfaces that do not exist.
        (["left_hand_controller"], ["--inactive"]),
    ]


def _spawners(pin_pelvis):
    return [
        ExecuteProcess(
            cmd=["ros2", "run", "controller_manager", "spawner", *names, *extra],
            name=f"{names[0]}_spawner",
            output="screen",
        )
        for names, extra in _controllers(pin_pelvis)
    ]


# The right Dex5 stays in the URDF with no component, but MoveIt's CurrentStateMonitor waits for a
# state on every non-mimic joint of the model before it plans. These are therefore reported at zero
# on /joint_states: a placeholder, not a measurement, and nothing can command them.
RIGHT_DEX5_JOINTS = [
    f"right_{joint}_joint"
    for joint in ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "little_1")
]


def _right_dex5_placeholder_state():
    message = (
        "{header: {stamp: now}, "
        f"name: [{', '.join(RIGHT_DEX5_JOINTS)}], "
        f"position: [{', '.join('0.0' for _ in RIGHT_DEX5_JOINTS)}]}}"
    )
    return ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "pub",
            "--rate",
            "10",
            "/joint_states",
            "sensor_msgs/msg/JointState",
            message,
        ],  # fmt: skip
        name="right_dex5_placeholder_state",
        output="log",
    )


def _robot_state_publisher():
    description = Command([FindExecutable(name="xacro"), " ", XACRO_PATH])
    return Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": ParameterValue(description, value_type=str)}],
    )


def _control_node():
    """`ros2 run` rather than launch_ros Node: under Node, arm_trajectory_controller's
    parameters reliably declare empty."""
    command = (
        "ros2 run controller_manager ros2_control_node --ros-args "
        "-r '~/robot_description:=/robot_description' "
        f"--params-file {CONTROLLERS_YAML}"
    )
    return ExecuteProcess(
        cmd=["bash", "-c", _SIGNAL_FORWARDING_WRAPPER.format(command=command)],
        name="ros2_control_node",
        output="screen",
    )


def _launch_setup(context, *args, **kwargs):
    # A welded pelvis gives the policy observations its own actions cannot move, so it diverges.
    pin_pelvis = LaunchConfiguration("pin_pelvis").perform(context).lower() in ("true", "1")
    control_node = _control_node()
    return [
        _robot_state_publisher(),
        control_node,
        *_spawners(pin_pelvis),
        _right_dex5_placeholder_state(),
        # Tear down the whole launch if controller_manager dies.
        RegisterEventHandler(
            OnProcessExit(
                target_action=control_node,
                on_exit=[EmitEvent(event=Shutdown(reason="ros2_control_node exited"))],
            )
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "pin_pelvis",
                default_value="false",
                description="SIM-ONLY, set by sim.launch.py: freeze the legs instead of running "
                "the balance policy.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
