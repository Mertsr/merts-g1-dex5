"""The RH56E2 hands, as they end up in the expanded description.

The vendor's files are SolidWorks exports with a few properties that are easy to lose on a
re-export and expensive to find later: a mimic chain that robot_state_publisher cannot resolve,
a joint that is driven but not declared, a hand mounted on the wrong frame. These tests read the
expanded URDF, which is what every consumer actually sees.
"""

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import pytest

SIDES = ("left", "right")
DRIVEN = ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "little_1")


@pytest.fixture(scope="module")
def robot():
    xacro_path = os.environ["G1_DESCRIPTION_XACRO"]
    with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as tmp:
        out_path = tmp.name
    subprocess.run(["xacro", xacro_path, "-o", out_path], check=True)
    root = ET.parse(out_path).getroot()
    os.remove(out_path)
    return root


def _hand_joints(robot, side):
    return [
        j for j in robot.findall("joint")
        if j.get("name").startswith(f"{side}_")
        and any(f in j.get("name") for f in ("thumb", "index", "middle", "ring", "little"))
        and "force_sensor" not in j.get("name")
    ]


def test_the_dex3_hands_are_gone(robot):
    # Two hands on one wrist would double the mass and fight over the frame. The derived
    # description is what removes them; a drift test keeps it honest, this checks the result.
    links = {link.get("name") for link in robot.findall("link")}
    assert not {name for name in links if "_hand_" in name and name.endswith("_link")} - {
        "left_hand_palm_link", "right_hand_palm_link"
    }


@pytest.mark.parametrize("side", SIDES)
def test_the_hand_is_mounted_on_the_palm_frame(robot, side):
    mount = next(j for j in robot.findall("joint") if j.get("name") == f"{side}_rh56e2_mount_joint")
    assert mount.get("type") == "fixed"
    assert mount.find("parent").get("link") == f"{side}_hand_palm_link"
    assert mount.find("child").get("link") == f"{side}_dex5_base_link"


@pytest.mark.parametrize("side", SIDES)
def test_exactly_six_joints_are_driven(robot, side):
    # Non-mimic revolute joints are what a controller must command. If a re-export drops a
    # <mimic> tag this grows past six; if it renames one, the set below stops matching.
    driven = {
        j.get("name") for j in _hand_joints(robot, side)
        if j.get("type") == "revolute" and j.find("mimic") is None
    }
    assert driven == {f"{side}_{name}_joint" for name in DRIVEN}


@pytest.mark.parametrize("side", SIDES)
def test_the_driven_joints_are_the_ones_ros2_control_declares(robot, side):
    component = next(
        c for c in robot.findall("ros2_control") if c.get("name") == f"G1HandSystem{side.capitalize()}"
    )
    declared = {j.get("name") for j in component.findall("joint")}
    driven = {
        j.get("name") for j in _hand_joints(robot, side)
        if j.get("type") == "revolute" and j.find("mimic") is None
    }
    assert declared == driven


@pytest.mark.parametrize("side", SIDES)
def test_no_mimic_follows_another_mimic(robot, side):
    # robot_state_publisher computes a mimic from the joint_states it receives, one level deep.
    # A mimic of a mimic reads a joint nobody publishes and stays at zero, which draws a thumb
    # that never bends at the tip. The vendor's export chains thumb_4 through thumb_3.
    joints = {j.get("name"): j for j in robot.findall("joint")}
    for joint in _hand_joints(robot, side):
        mimic = joint.find("mimic")
        if mimic is not None:
            assert joints[mimic.get("joint")].find("mimic") is None, (
                f"{joint.get('name')} mimics {mimic.get('joint')}, which is itself a mimic"
            )


@pytest.mark.parametrize("side", SIDES)
def test_every_range_starts_open_at_zero_and_closes_positive(robot, side):
    # The SRDF's `open` posture is all zeros and `closed` is all positive, on both hands. That
    # only holds while this does.
    for joint in _hand_joints(robot, side):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        assert float(limit.get("lower")) == 0.0, joint.get("name")
        assert float(limit.get("upper")) > 0.0, joint.get("name")


@pytest.mark.parametrize("side", SIDES)
def test_the_grasp_frame_hangs_off_the_palm_frame(robot, side):
    joint = next(j for j in robot.findall("joint") if j.get("name") == f"{side}_hand_grasp_frame_joint")
    assert joint.find("parent").get("link") == f"{side}_hand_palm_link"
    assert joint.find("child").get("link") == f"{side}_hand_grasp_frame"


@pytest.mark.parametrize("side", SIDES)
def test_the_hand_meshes_resolve_under_this_package(robot, side):
    # Paths only: the STLs are checked in separately and may not be present in every checkout.
    prefix = f"package://g1_description/meshes/rh56e2/{side}/"
    meshes = [
        m.get("filename") for link in robot.findall("link")
        if link.get("name").startswith(f"{side}_") and "hand" not in link.get("name")
        for m in link.iter("mesh")
        if "rh56e2" in m.get("filename")
    ]
    assert meshes, f"no RH56E2 meshes found for the {side} hand"
    assert all(m.startswith(prefix) for m in meshes)
