# Validates the whole-body xacro: the body component and its wire order.
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import pytest

# SDK motor order, 0-28. The component packs LowCmd.motor_cmd positionally from this.
EXPECTED_BODY_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _expand(xacro_path):
    with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as tmp:
        out_path = tmp.name
    subprocess.run(["xacro", xacro_path, "-o", out_path], check=True)
    return out_path


@pytest.fixture(scope="module")
def expanded_urdf_path():
    path = _expand(os.environ["G1_LOWCMD_XACRO"])
    yield path
    os.remove(path)


def component(root, name):
    return next(c for c in root.findall("ros2_control") if c.get("name") == name)


def test_xacro_expands_to_valid_urdf(expanded_urdf_path):
    result = subprocess.run(["check_urdf", expanded_urdf_path], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_component_claims_all_29_body_motors_in_sdk_order(expanded_urdf_path):
    root = ET.parse(expanded_urdf_path).getroot()
    body = component(root, "G1LowCmdSystem")
    assert body.find("hardware/plugin").text == "g1_hardware_interface/G1LowCmdSystem"
    assert [j.get("name") for j in body.findall("joint")] == EXPECTED_BODY_JOINTS


def test_every_body_joint_exports_kp_and_kd(expanded_urdf_path):
    # kp and kd as command interfaces let a controller pick the joint's control mode per tick;
    # without them the policy cannot send the gains it infers.
    root = ET.parse(expanded_urdf_path).getroot()
    for joint in component(root, "G1LowCmdSystem").findall("joint"):
        commands = [c.get("name") for c in joint.findall("command_interface")]
        assert commands == ["position", "velocity", "effort", "kp", "kd"], joint.get("name")
        params = {p.get("name") for p in joint.findall("param")}
        assert {"position_only_kp", "position_only_kd"} <= params, joint.get("name")


def test_no_hand_ros2_control_component(expanded_urdf_path):
    # The Dex5 hands are description-only for now; the old Dex3 components are gone with their
    # joints.
    root = ET.parse(expanded_urdf_path).getroot()
    assert [c.get("name") for c in root.findall("ros2_control")] == ["G1LowCmdSystem"]
