"""g1_29dof_no_hands.urdf is derived, and a derived file that has drifted is a silent bug.

It is generated from the vendored Unitree description by scripts/make_no_hands_urdf.py. If
someone edits it by hand, or updates the vendored file and forgets to regenerate, the model
the rest of the stack plans against stops matching what the script would produce.
"""

import pathlib
import subprocess
import sys

_PACKAGE = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT = _PACKAGE / "scripts" / "make_no_hands_urdf.py"


def test_the_checked_in_file_matches_what_the_script_derives():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_nothing_of_the_dex3_remains_but_the_two_empty_palm_frames():
    text = (_PACKAGE / "urdf" / "g1_29dof_no_hands.urdf").read_text()
    for side in ("left", "right"):
        assert f'<link name="{side}_hand_palm_link"/>' in text
        for finger in ("thumb", "middle", "index"):
            for n in range(3):
                assert f"{side}_hand_{finger}_{n}_" not in text
