# Structural checks of the bilateral Dex5 (RH56E2) hands in the expanded description. Needs only
# xacro; check_urdf is run when present.
import pathlib
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

_URDF_DIR = pathlib.Path(__file__).resolve().parent.parent / "urdf"
_XACRO = _URDF_DIR / "g1_lowcmd.urdf.xacro"
_SOURCES = {"left": "RH56E2_L_2025_9_10.xacro", "right": "RH56E2_R_2025_9_11.xacro"}
_ACTIVE = ["thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "little_1"]
_MIMIC = {
    "thumb_3": "thumb_2",
    "thumb_4": "thumb_3",
    "index_2": "index_1",
    "middle_2": "middle_1",
    "ring_2": "ring_1",
    "little_2": "little_1",
}


@pytest.fixture(scope="module")
def robot(tmp_path_factory):
    out = tmp_path_factory.mktemp("dex5") / "g1.urdf"
    subprocess.run(["xacro", str(_XACRO), "-o", str(out)], check=True)
    return out, ET.parse(out).getroot()


def test_names_unique(robot):
    _, root = robot
    for tag in ("link", "joint"):
        names = [e.get("name") for e in root.findall(tag)]
        assert len(names) == len(set(names)), tag


def test_tree_is_closed_and_connected(robot):
    _, root = robot
    links = {link.get("name") for link in root.findall("link")}
    children = {}
    for j in root.findall("joint"):
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        assert parent in links and child in links, j.get("name")
        assert child not in children, f"{child} has two parents"
        children[child] = parent
    roots = links - children.keys()
    assert len(roots) == 1
    for link in links:
        while link in children:
            link = children[link]
        assert link in roots


def test_mimic_references_exist(robot):
    _, root = robot
    joints = {j.get("name") for j in root.findall("joint")}
    for j in root.findall("joint"):
        m = j.find("mimic")
        if m is not None:
            assert m.get("joint") in joints, j.get("name")


@pytest.mark.parametrize("side", ["left", "right"])
def test_wrist_mount_and_hand_joints(robot, side):
    _, root = robot
    joints = {j.get("name"): j for j in root.findall("joint")}
    mount = joints[f"{side}_dex5_mount_joint"]
    assert mount.get("type") == "fixed"
    assert mount.find("parent").get("link") == f"{side}_wrist_yaw_link"
    assert mount.find("child").get("link") == f"{side}_dex5_base_link"

    for f in _ACTIVE:
        j = joints[f"{side}_{f}_joint"]
        assert j.get("type") == "revolute" and j.find("mimic") is None, f
    for f, target in _MIMIC.items():
        m = joints[f"{side}_{f}_joint"].find("mimic")
        assert m is not None and m.get("joint") == f"{side}_{target}_joint", f
    assert f"{side}_plam_1_joint" in joints
    assert f"{side}_plam_force_sensor_joint" in joints


def test_old_unitree_hand_chains_removed(robot):
    _, root = robot
    for e in root.findall("link") + root.findall("joint"):
        assert "_hand_" not in e.get("name") or "grasp_frame" in e.get("name"), e.get("name")


@pytest.mark.parametrize("side", ["left", "right"])
def test_mesh_uris(robot, side):
    _, root = robot
    prefix = f"package://g1_description/meshes/dex5_{side}/"
    meshes = [
        m.get("filename")
        for link in root.findall("link")
        for m in link.iter("mesh")
        if m.get("filename").startswith(prefix)
    ]
    assert len(meshes) == 64
    assert all(m.endswith(".STL") for m in meshes)


def test_sources_have_no_stale_package_names():
    for src in _SOURCES.values():
        assert "RH56E2_" not in (_URDF_DIR / src).read_text().replace(src.split(".")[0], "")


def test_host_urdf_is_plain():
    assert "xacro" not in (_URDF_DIR / "g1_29dof_with_hand_rev_1_0.urdf").read_text()


@pytest.mark.skipif(shutil.which("check_urdf") is None, reason="check_urdf not installed")
def test_check_urdf(robot):
    path, _ = robot
    result = subprocess.run(["check_urdf", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
