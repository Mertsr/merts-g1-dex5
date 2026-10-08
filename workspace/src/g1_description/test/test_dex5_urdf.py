"""Structural checks for the bilateral Dex5 hands and their URDF wrapper flow."""
import pathlib
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

_URDF_DIR = pathlib.Path(__file__).resolve().parent.parent / "urdf"
_HOST_URDF = _URDF_DIR / "g1_29dof_with_hand_rev_1_0.urdf"
_COMMON_XACRO = _URDF_DIR / "g1_common.xacro"
_LOWCMD_XACRO = _URDF_DIR / "g1_lowcmd.urdf.xacro"
_SOURCES = {
    "left": "RH56E2_L_2025_9_10.xacro",
    "right": "RH56E2_R_2025_9_11.xacro",
}
_ACTIVE = ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "little_1")
_MIMIC_MULTIPLIERS = {
    "thumb_3": ("thumb_2", 0.8392),
    "thumb_4": ("thumb_3", 0.891),
    "index_2": ("index_1", 1.0843),
    "middle_2": ("middle_1", 1.0843),
    "ring_2": ("ring_1", 1.0843),
    "little_2": ("little_1", 1.0843),
}


def _names(root, tag):
    return [element.get("name") for element in root.findall(tag)]


def _assert_unique_names(root):
    for tag in ("link", "joint"):
        names = _names(root, tag)
        assert None not in names
        assert len(names) == len(set(names)), tag


def _assert_tree_is_closed_and_connected(root):
    links = set(_names(root, "link"))
    children = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        assert parent in links and child in links, joint.get("name")
        assert child not in children, f"{child} has two parents"
        children[child] = parent
    roots = links - children.keys()
    assert roots == {"pelvis"}
    for link in links:
        while link in children:
            link = children[link]
        assert link in roots


def _normalized_xml(element):
    element = ET.fromstring(ET.tostring(element, encoding="unicode"))
    for descendant in element.iter():
        descendant.tail = None
    return ET.tostring(element, encoding="unicode")


def _source_root(side):
    return ET.parse(_URDF_DIR / _SOURCES[side]).getroot()


def _validate_hands(root):
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    links = set(_names(root, "link"))
    for side in _SOURCES:
        mount = joints[f"{side}_dex5_mount_joint"]
        assert mount.get("type") == "fixed"
        assert mount.find("parent").get("link") == f"{side}_wrist_yaw_link"
        assert mount.find("child").get("link") == f"{side}_dex5_base_link"
        origin = mount.find("origin")
        assert origin.get("xyz") == "0 0 0" and origin.get("rpy") == "0 0 0"

        assert len([link for link in links if link.startswith(f"{side}_")]) > 0
        for finger in _ACTIVE:
            joint = joints[f"{side}_{finger}_joint"]
            assert joint.get("type") == "revolute"
            assert joint.find("mimic") is None
        for finger, (target, multiplier) in _MIMIC_MULTIPLIERS.items():
            mimic = joints[f"{side}_{finger}_joint"].find("mimic")
            assert mimic is not None
            assert mimic.get("joint") == f"{side}_{target}_joint"
            assert float(mimic.get("multiplier")) == pytest.approx(multiplier)
            assert float(mimic.get("offset")) == 0

        source_joint_names = {joint.get("name") for joint in _source_root(side).findall("joint")}
        hand_joints = [joints[name] for name in source_joint_names]
        assert sum(joint.get("type") == "revolute" for joint in hand_joints) == 12
        assert sum(joint.get("type") == "fixed" for joint in hand_joints) == 19
        assert f"{side}_plam_1" in links
        assert f"{side}_plam_force_sensor" in links
        assert f"{side}_hand_grasp_frame" in links

        prefix = f"package://g1_description/meshes/dex5_{side}/"
        mesh_uris = [
            mesh.get("filename")
            for link in root.findall("link")
            for mesh in link.iter("mesh")
            if mesh.get("filename", "").startswith(prefix)
        ]
        assert len(mesh_uris) == 64
        assert all(uri.endswith(".STL") for uri in mesh_uris)


def test_host_urdf_is_plain_well_formed_urdf():
    raw = _HOST_URDF.read_text()
    assert not re.search(r"<\s*xacro:|xmlns:xacro|\$\(\s*arg|\$\{", raw)
    root = ET.fromstring(raw)
    assert root.tag == "robot"
    assert root.attrib == {"name": "g1_29dof_with_hand_rev_1_0"}


def test_host_urdf_contains_exact_source_hand_models():
    host = ET.parse(_HOST_URDF).getroot()
    _assert_unique_names(host)
    _assert_tree_is_closed_and_connected(host)
    _validate_hands(host)

    host_links = {link.get("name"): link for link in host.findall("link")}
    host_joints = {joint.get("name"): joint for joint in host.findall("joint")}
    for side in _SOURCES:
        source = _source_root(side)
        assert len(source.findall("link")) == 32
        assert len(source.findall("joint")) == 31
        for tag, host_elements in (("link", host_links), ("joint", host_joints)):
            for element in source.findall(tag):
                name = element.get("name")
                assert name in host_elements
                assert _normalized_xml(element) == _normalized_xml(host_elements[name])


def test_unitree_hand_chains_are_removed():
    root = ET.parse(_HOST_URDF).getroot()
    names = _names(root, "link") + _names(root, "joint")
    for name in names:
        assert not re.search(r"(?:left|right)_hand_(?:palm|thumb|index|middle|ring|little)", name), name
        if "_hand_" in name:
            assert name in {
                "left_hand_grasp_frame", "left_hand_grasp_frame_joint",
                "right_hand_grasp_frame", "right_hand_grasp_frame_joint",
            }


def test_common_xacro_includes_host_once_and_keeps_sensor_frames():
    common = _COMMON_XACRO.read_text()
    assert common.count('filename="g1_29dof_with_hand_rev_1_0.urdf"') == 1
    assert "RH56E2_L_2025_9_10.xacro" not in common
    assert "RH56E2_R_2025_9_11.xacro" not in common
    assert "dex5_left_mount" not in common and "dex5_right_mount" not in common
    for frame in ("camera_depth_optical_frame", "camera_color_optical_frame",
                  "chest_camera_link", "mid360_imu"):
        assert f'name="{frame}"' in common


@pytest.mark.skipif(shutil.which("xacro") is None, reason="xacro not installed")
def test_lowcmd_wrapper_expands_without_duplicate_hands(tmp_path):
    output = tmp_path / "g1.urdf"
    subprocess.run(["xacro", str(_LOWCMD_XACRO), "-o", str(output)], check=True)
    root = ET.parse(output).getroot()
    _assert_unique_names(root)
    _assert_tree_is_closed_and_connected(root)
    _validate_hands(root)
    assert [component.get("name") for component in root.findall("ros2_control")] == [
        "G1LowCmdSystem"
    ]
    for frame in ("camera_depth_optical_frame", "camera_color_optical_frame",
                  "chest_camera_link", "mid360_imu"):
        assert frame in set(_names(root, "link"))
