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


EXPECTED_LEFT_DEX5_JOINTS = [
    "left_thumb_1_joint",
    "left_thumb_2_joint",
    "left_index_1_joint",
    "left_middle_1_joint",
    "left_ring_1_joint",
    "left_little_1_joint",
]
MIMIC_JOINTS = [
    "left_thumb_3_joint",
    "left_thumb_4_joint",
    "left_index_2_joint",
    "left_middle_2_joint",
    "left_ring_2_joint",
    "left_little_2_joint",
]


def test_only_the_body_and_the_left_dex5_hand_have_components(expanded_urdf_path):
    root = ET.parse(expanded_urdf_path).getroot()
    assert [c.get("name") for c in root.findall("ros2_control")] == [
        "G1LowCmdSystem",
        "G1Dex5SystemLeft",
    ]


def test_body_component_is_still_exactly_29_joints(expanded_urdf_path):
    root = ET.parse(expanded_urdf_path).getroot()
    assert len(component(root, "G1LowCmdSystem").findall("joint")) == 29


def test_left_hand_claims_exactly_the_six_active_joints_in_order(expanded_urdf_path):
    root = ET.parse(expanded_urdf_path).getroot()
    hand = component(root, "G1Dex5SystemLeft")
    assert [j.get("name") for j in hand.findall("joint")] == EXPECTED_LEFT_DEX5_JOINTS
    claimed = {j.get("name") for c in root.findall("ros2_control") for j in c.findall("joint")}
    assert not claimed & set(MIMIC_JOINTS)
    assert not [n for n in claimed if n.startswith("right_") and n.endswith("_1_joint")]
    # No hand joint is claimed twice, or by the body component too.
    body = {j.get("name") for j in component(root, "G1LowCmdSystem").findall("joint")}
    assert not body & set(EXPECTED_LEFT_DEX5_JOINTS)


def test_left_hand_joints_export_position_command_and_pve_state(expanded_urdf_path):
    root = ET.parse(expanded_urdf_path).getroot()
    for joint in component(root, "G1Dex5SystemLeft").findall("joint"):
        commands = [c.get("name") for c in joint.findall("command_interface")]
        states = [c.get("name") for c in joint.findall("state_interface")]
        assert commands == ["position"], joint.get("name")
        assert states == ["position", "velocity", "effort"], joint.get("name")


def test_left_hand_limit_params_match_the_urdf(expanded_urdf_path):
    root = ET.parse(expanded_urdf_path).getroot()
    urdf_limits = {j.get("name"): j.find("limit") for j in root.findall("joint")}
    for joint in component(root, "G1Dex5SystemLeft").findall("joint"):
        params = {p.get("name"): float(p.text) for p in joint.findall("param")}
        limit = urdf_limits[joint.get("name")]
        assert params["min"] == pytest.approx(float(limit.get("lower")), abs=1e-6)
        assert params["max"] == pytest.approx(float(limit.get("upper")), abs=1e-6)


def test_the_simulation_backend_is_selected_without_dds(expanded_urdf_path):
    root = ET.parse(expanded_urdf_path).getroot()
    hand = component(root, "G1Dex5SystemLeft")
    assert hand.find("hardware/plugin").text == "g1_hand_interface/G1Dex5SimSystem"
    params = {p.get("name") for p in hand.findall("hardware/param")}
    assert not params & {"domain_id", "network_interface", "state_topic"}
    assert "Dex3" not in ET.tostring(hand, encoding="unicode")


def test_an_unknown_dex5_backend_fails_the_expansion():
    result = subprocess.run(
        ["xacro", os.environ["G1_LOWCMD_XACRO"], "dex5_backend:=bogus"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
