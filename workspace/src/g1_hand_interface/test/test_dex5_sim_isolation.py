"""The Dex5 simulation startup path must not reach the Dex3 SDK component or open DDS channels.

Static checks over the sources and configs the path is made of; no ROS graph needed.
"""

import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent.parent.parent
FORBIDDEN = ("G1Dex3System", "unitree/", "unitree_sdk2", "ChannelFactory", "ChannelPublisher")


def _read(*parts):
    return (SRC.joinpath(*parts)).read_text()


def test_the_sim_backend_sources_do_not_touch_the_sdk_or_dex3():
    for path in (
        ("g1_hand_interface", "src", "g1_dex5_sim_system.cpp"),
        ("g1_hand_interface", "include", "g1_hand_interface", "g1_dex5_sim_system.hpp"),
        ("g1_hand_interface", "include", "g1_hand_interface", "dex5_sim_model.hpp"),
    ):
        text = _read(*path)
        for token in FORBIDDEN:
            assert token not in text, f"{path[-1]} mentions {token}"


def test_the_sim_library_does_not_link_the_sdk():
    cmake = _read("g1_hand_interface", "CMakeLists.txt")
    block = re.search(r"add_library\(g1_dex5_sim_system.*?\n\)", cmake, re.S).group(0)
    assert "unitree_sdk2" not in block
    assert "g1_dex3_system" not in block


def test_the_sim_plugin_is_exported_under_its_own_class():
    xml = _read("g1_hand_interface", "g1_dex5_sim.xml")
    assert 'name="g1_hand_interface/G1Dex5SimSystem"' in xml
    assert "Dex3" not in xml


def test_the_active_startup_path_names_no_dex3_component():
    for path in (
        ("g1_description", "urdf", "g1_lowcmd.urdf.xacro"),
        ("g1_description", "config", "dex5_params.yaml"),
        ("g1_controllers", "config", "lowcmd_controllers.yaml"),
        ("g1_bringup", "launch", "control.launch.py"),
        ("g1_bringup", "scripts", "activate_arm"),
        ("g1_bringup", "scripts", "deactivate_arm"),
        ("g1_orchestration", "src", "arm_authority.cpp"),
    ):
        text = _read(*path)
        assert "G1Dex3System" not in text, f"{path[-1]} still names G1Dex3System"
        assert "rt/dex3" not in text, f"{path[-1]} still names a Dex3 channel"
