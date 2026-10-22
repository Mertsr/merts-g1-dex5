#!/usr/bin/env python3
"""Generate 003-add-rh56e2-hands.patch from the expanded G1 description.

The simulated G1 has to end its arms in the same hands MoveIt plans for, or the simulator and
the model drift. Hand-editing a patch against an MJCF is how they would drift, so the patch is
generated from the URDF instead, and this is the generator. Run it whenever the hand
description (g1_description/urdf/rh56e2_*.urdf.xacro, config/rh56e2_params.yaml) changes.

    xacro workspace/src/g1_description/urdf/g1_arm_sdk.urdf.xacro -o /tmp/g1.urdf
    git -C <unitree_mujoco at the pinned SHA, with patches 001 and 002 applied> ... (see below)
    make_rh56e2_patch.py --urdf /tmp/g1.urdf --mjcf <that tree>/unitree_robots/g1/g1_29dof.xml \\
        --out workspace/patches/unitree_mujoco/003-add-rh56e2-hands.patch

`--mjcf` must be the file as it is after patches 001 and 002 and before 003; those two do not
touch it, so the pristine upstream file works too. The output is a unified diff against it.

What it does to g1_29dof.xml:
  * replaces each arm's fixed rubber-hand geom with the RH56E2 body tree, walked out of the
    URDF from `{side}_hand_palm_link` down;
  * adds a joint-equality constraint per URDF mimic joint, so the passive finger segments
    follow their drivers;
  * puts the wrist links' inertials back to the URDF's own, since the stock model lumped the
    rubber hand's mass into them;
  * gives every hand body a collision geom (its mesh's convex hull) so the hand stops at walls
    and can hold objects by friction, and excludes the flange-mounted bodies from colliding
    with the wrist links they sit against;
  * scales the hand to a simulator-only mass. See the patch header, which is written here.
"""

import argparse
import difflib
import math
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

# The mass the stock model lumped into each wrist_yaw_link for its rubber hand. Holding each
# hand to this keeps the model's total mass and its distribution where they were, which is what
# the stand-in walking policy needs. The real hand weighs more; see PATCH_HEADER.
SIM_HAND_MASS_KG = 0.170

SIDES = ("left", "right")
WRIST = "{side}_wrist_yaw_link"
PALM = "{side}_hand_palm_link"

PATCH_HEADER = """\
Gives the simulated G1 the Inspire RH56E2 hands.

Generated against unitree_mujoco ae6a8403e272733e9996ef59990880330496177f by
workspace/patches/unitree_mujoco/tools/make_rh56e2_patch.py, from the expanded
g1_description/urdf/g1_arm_sdk.urdf.xacro, so the simulator and the model MoveIt plans against
cannot drift. Do not edit by hand; rerun the generator. It replaces 003-add-dex3-hands.patch.

The stock g1_29dof.xml ends each arm in a fixed rubber-hand geom with no joints. This replaces
both with the real body tree: {bodies} bodies per hand, {joints} hinge joints, of which {driven} are
driven and {mimic} follow a driver through a joint equality constraint (the URDF's <mimic> tags,
with the same multipliers). The meshes are loaded from meshes/rh56e2/{{left,right}}/, which the
image build copies in from g1_description/meshes/rh56e2/.

THE HAND COLLIDES. Every hand body carries a second geom of class rh56e2_collision: the same
mesh, collided as its convex hull, hidden in group 3. It contacts the world, scene objects and
the rest of the robot, never another hand geom (bitmask, see the class), and the bodies bolted
to the flange are excluded from the two wrist links. So the hand stops at a wall instead of
passing through it, and a closed hand holds an object by friction. grasp_weld still runs when
its config enables it; with contact on, a hand that closes on a large object stops on its
surface and may never reach the weld's closure threshold, so the grip is then friction alone.
The fingertip force-sensor pads get no collision geom: g1_description still ships them as 1 cm
placeholder tetrahedra, which as hulls stuck out of the fingertips as spikes and were the only
thing touching a part. The finger segments' own meshes collide there instead.

NO ACTUATORS AND NO SENSORS, which is not an oversight. RobotBridge sets
num_motor_ = mj_model_->nu and then indexes lowcmd->msg_.motor_cmd()[i], a fixed
std::array<MotorCmd_, 35>; it also reads motor state as three contiguous nu-sized blocks
of sensordata. Adding {driven2} actuators would take nu to {nu} and run both loops off the end of
those arrays. Keeping nu at 29 leaves the vendored bridge byte-identical and correct, and
the hand is a separate device on its own DDS topics anyway, so it
is driven through qfrc_applied by dex3_handler instead (the file keeps its Dex3 name; it now
serves the RH56E2).

HAND MASS IS SCALED TO {mass:.3f} kg PER HAND, not the real {real:.3f} kg, and that is a deliberate
simulator-only deviation. {mass:.3f} kg is what the stock model lumped into each wrist_yaw_link
for the rubber hand, so the model's total mass and its distribution come out exactly where
they were. Each wrist_yaw_link inertial goes back to the URDF's own value, since the hand is now
a body of its own and the lumped mass would double-count.

Why: the walking policy in g1_motion_service_sim is a third-party RL checkpoint, and it is
this stack's stand-in for a controller the real robot has and the simulator does not. It could
not balance under the Dex3's true hand mass (0.697 kg), measured: 6 of 12 balance tests failing.
The RH56E2 is heavier still. On hardware locomotion goes through LocoClient to Unitree's own
onboard controller, which never sees this policy, so the instability is structurally
simulator-only. What it costs: the simulated arm carries less inertia than the real one, so arm
motion under arm_sdk will track better here than on hardware.

THE POLICY HAS NOT BEEN RE-TESTED WITH THIS HAND. The scaling keeps the mass and its position
along the arm as they were, but the RH56E2 is much longer than the rubber hand, so its centre of
mass sits further out, and the stock scaling argument covers mass and not that. Run
g1_motion_service_sim's balance tests before trusting it.
"""


# ---------------------------------------------------------------- small linear algebra
def rpy_to_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def matrix_to_quat(m):
    """Rotation matrix to (w, x, y, z), with w >= 0."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = (
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w, x, y, z = (
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        )
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w, x, y, z = (
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        )
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w, x, y, z = (
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        )
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return w / n, x / n, y / n, z / n


def num(value):
    """Short, stable number formatting; -0 becomes 0."""
    text = f"{value:.9g}"
    return "0" if text in ("-0", "0") else text


def vec(values):
    return " ".join(num(v) for v in values)


def floats(text, default="0 0 0"):
    return [float(v) for v in (text or default).split()]


# ---------------------------------------------------------------- URDF reading
class Urdf:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.links = {link.get("name"): link for link in root.findall("link")}
        self.joints = {j.get("name"): j for j in root.findall("joint")}
        self.children = {}
        for j in self.joints.values():
            self.children.setdefault(j.find("parent").get("link"), []).append(j)

    def inertial(self, link):
        node = self.links[link].find("inertial")
        if node is None:
            return None
        origin = node.find("origin")
        if origin is not None and any(abs(v) > 0 for v in floats(origin.get("rpy"))):
            raise SystemExit(f"{link}: a rotated inertial frame is not handled")
        i = node.find("inertia")
        return {
            "pos": floats(origin.get("xyz") if origin is not None else None),
            "mass": float(node.find("mass").get("value")),
            "I": [float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")],
        }

    def visual(self, link):
        node = self.links[link].find("visual")
        if node is None:
            return None
        mesh = node.find("geometry/mesh")
        rgba = node.find("material/color")
        return {
            "file": mesh.get("filename").split("/meshes/", 1)[1],
            "rgba": floats(rgba.get("rgba"), "0.7 0.7 0.7 1")
            if rgba is not None
            else [0.7, 0.7, 0.7, 1],
        }


def collect_hand(urdf, side):
    """Walk the hand out of the URDF, depth first, from the palm frame."""
    palm = PALM.format(side=side)
    palm_joint = next(j for j in urdf.joints.values() if j.find("child").get("link") == palm)
    bodies = []  # (joint element, parent link) in document order

    def walk(link):
        for joint in urdf.children.get(link, []):
            # The grasp and pinch frames are URDF-only planning frames; the sim welds to the palm.
            if joint.find("child").get("link").endswith(("_hand_grasp_frame", "_hand_pinch_frame")):
                continue
            bodies.append(joint)
            walk(joint.find("child").get("link"))

    walk(palm)
    return palm_joint, bodies


# ---------------------------------------------------------------- MJCF emission
def emit_body(urdf, joint, scale, depth, lines, joints_out, mesh_out, rigid_out, rigid=True):
    """Emits one hand body and its subtree.

    `rigid` is True while no revolute joint lies between this body and the palm frame; those
    bodies are bolted to the wrist flange and are collected in `rigid_out` so the wrist can be
    excluded from colliding with them.
    """
    child = joint.find("child").get("link")
    origin = joint.find("origin")
    xyz = floats(origin.get("xyz") if origin is not None else None)
    rpy = floats(origin.get("rpy") if origin is not None else None)
    quat = matrix_to_quat(rpy_to_matrix(*rpy))
    pad = "  " * depth

    attrs = f'name="{child}" pos="{vec(xyz)}"'
    if any(abs(v) > 1e-12 for v in rpy):
        attrs += f' quat="{vec(quat)}"'
    lines.append(f"{pad}<body {attrs}>")

    inertial = urdf.inertial(child)
    if inertial is not None:
        i = [v * scale for v in inertial["I"]]
        lines.append(
            f'{pad}  <inertial pos="{vec(inertial["pos"])}" mass="{num(inertial["mass"] * scale)}" '
            f'fullinertia="{vec(i)}" />'
        )

    if joint.get("type") == "revolute":
        rigid = False
        limit = joint.find("limit")
        axis = floats(joint.find("axis").get("xyz"))
        norm = math.sqrt(sum(a * a for a in axis))
        axis = [a / norm for a in axis]
        lines.append(
            f'{pad}  <joint name="{joint.get("name")}" axis="{vec(axis)}" '
            f'range="{num(float(limit.get("lower")))} {num(float(limit.get("upper")))}" '
            f'class="rh56e2_motor" />'
        )
        joints_out.append(joint)

    visual = urdf.visual(child)
    if visual is not None:
        lines.append(
            f'{pad}  <geom class="rh56e2_visual" mesh="{child}" rgba="{vec(visual["rgba"])}" />'
        )
        # Same mesh, collided as its convex hull. Separate from the visual geom so the visual
        # keeps group 1 and the collision geom can be hidden (group 3) and tuned on its own.
        # Not for the fingertip force-sensor pads: g1_description ships them as 1 cm placeholder
        # tetrahedra, whose hulls stick out of the fingertips as spikes and become the only
        # thing touching a part -- point contacts at odd angles that let any grip slide off.
        # The real pads lie flush on the finger, whose own mesh already collides there.
        if "_force_sensor" not in child:
            lines.append(f'{pad}  <geom class="rh56e2_collision" mesh="{child}" />')
        mesh_out.append((child, visual["file"]))
        if rigid:
            rigid_out.append(child)

    for nxt in urdf.children.get(child, []):
        emit_body(urdf, nxt, scale, depth + 1, lines, joints_out, mesh_out, rigid_out, rigid)
    lines.append(f"{pad}</body>")


def build(urdf, mjcf_text):
    stats = {}
    equalities = []
    excludes = []
    text = mjcf_text

    # -- defaults
    anchor = '      <joint damping="0.05" armature="0.01" frictionloss="0.1"/>\n    </default>\n    \n  </default>\n'
    if text.count(anchor) != 1:
        raise SystemExit("the upstream <default> block is not shaped as expected")
    defaults = (
        '      <joint damping="0.05" armature="0.01" frictionloss="0.1"/>\n    </default>\n'
        "    <!-- RH56E2 fingers. No <actuator> and no <sensor> entries on purpose: the SDK bridge\n"
        "         sizes itself from nu and indexes a fixed 35-slot LowCmd, so 29 + 12 would run it\n"
        "         off the end of the array. The hand is a separate device on its own topics and is\n"
        "         driven through qfrc_applied instead. armature deliberately dominates the physical\n"
        "         finger inertia (~1e-5 kg m^2), which keeps the hand's PD overdamped at the 2 ms\n"
        "         timestep so its gains drive the sim unchanged rather than being retuned for it. -->\n"
        '    <default class="rh56e2_motor">\n'
        '      <joint damping="0.01" armature="0.001" frictionloss="0.01"/>\n'
        "    </default>\n"
        "    <!-- What you see: no contact, no mass (the bodies carry explicit inertials). -->\n"
        '    <default class="rh56e2_visual">\n'
        '      <geom type="mesh" contype="0" conaffinity="0" group="1" density="0"\n'
        '        rgba="0.7 0.7 0.7 1"/>\n'
        "    </default>\n"
        "    <!-- What collides: the same meshes as convex hulls, hidden in group 3. Bitmask\n"
        "         contype=2 / conaffinity=1 against everything else's 1 / 1: a hand geom contacts the\n"
        "         world, scene objects and the rest of the robot, but never another hand geom --\n"
        "         adjacent finger hulls overlap at their joints and would otherwise be in permanent\n"
        "         contact. condim 4 adds torsional friction so a pinched object does not spin out;\n"
        "         sliding friction 1.5 is a GUESS for rubber finger pads, not a measurement. -->\n"
        '    <default class="rh56e2_collision">\n'
        '      <geom type="mesh" contype="2" conaffinity="1" condim="4" group="3" density="0"\n'
        '        friction="1.5 0.02 0.001" rgba="0.9 0.4 0.3 0.5"/>\n'
        "    </default>\n"
        "  </default>\n"
    )
    text = text.replace(anchor, defaults)

    for side in SIDES:
        wrist = WRIST.format(side=side)
        palm_joint, chain = collect_hand(urdf, side)
        top = [j for j in chain if j.find("parent").get("link") == PALM.format(side=side)]
        assert len(top) == 1, f"{side}: expected one body on the palm frame"

        # total mass of the subtree, to scale it to the simulator's budget
        links = [j.find("child").get("link") for j in chain]
        real_mass = sum(urdf.inertial(link)["mass"] for link in links if urdf.inertial(link))
        scale = SIM_HAND_MASS_KG / real_mass
        stats["real_mass"] = real_mass

        # -- palm frame (empty) with the hand below it
        origin = palm_joint.find("origin")
        pxyz = floats(origin.get("xyz"))
        lines = []
        joints_out, meshes_out, rigid_out = [], [], []
        emit_body(urdf, top[0], scale, 0, lines, joints_out, meshes_out, rigid_out)
        # The hand's flange-mounted bodies sit flush against the wrist links, so their hulls
        # touch or overlap them at rest. Neither wrist link is the hand's direct parent (the
        # empty palm frame is), so MuJoCo's parent-child filter does not cover them.
        for wrist_link in (WRIST.format(side=side), f"{side}_wrist_pitch_link"):
            for body_name in rigid_out:
                excludes.append(f'    <exclude body1="{wrist_link}" body2="{body_name}" />\n')
        body = "\n".join("                            " + ln for ln in lines)
        stats["bodies"] = sum(1 for ln in lines if ln.lstrip().startswith("<body "))
        stats["joints"] = len(joints_out)
        stats["mimic"] = sum(1 for j in joints_out if j.find("mimic") is not None)
        stats["driven"] = stats["joints"] - stats["mimic"]

        block = (
            f"                          <!-- Inspire RH56E2, generated from the expanded g1_arm_sdk.urdf.xacro by\n"
            f"                               make_rh56e2_patch.py so the sim and the model MoveIt plans\n"
            f"                               against cannot drift. {side}_hand_palm_link is an empty frame at the\n"
            f"                               flange, as in the URDF, and is what grasp_weld welds to. Masses are\n"
            f"                               scaled to {SIM_HAND_MASS_KG:.3f} kg per hand; the patch header says why.\n"
            f"                               The wrist inertial above is the URDF's own again. -->\n"
            f'                          <body name="{PALM.format(side=side)}" pos="{vec(pxyz)}">\n'
            f"{body}\n"
            f"                          </body>\n"
        )

        # -- the wrist: URDF inertial back, rubber hand out
        pattern = re.compile(
            r'(?P<head>                        <body name="' + wrist + r'" pos="[^"]+">\n)'
            r"                          <inertial [^>]*?/>\n"
            r"(?P<mid>.*?)"
            r'                          <geom pos="[^"]*" quat="[^"]*" type="mesh" contype="0"\n'
            r'                            conaffinity="0" group="1" density="0" rgba="0.7 0.7 0.7 1"\n'
            r'                            mesh="' + side + r'_rubber_hand" />\n',
            re.S,
        )
        wi = urdf.inertial(wrist)
        wrist_inertial = (
            f'                          <inertial pos="{vec(wi["pos"])}" mass="{num(wi["mass"])}" '
            f'fullinertia="{vec(wi["I"])}" />\n'
        )
        text, n = pattern.subn(
            lambda m, wi_=wrist_inertial, blk=block: m["head"] + wi_ + m["mid"] + blk, text
        )
        if n != 1:
            raise SystemExit(f"{side}: could not find the {wrist} body with its rubber hand")

        # -- assets
        asset_lines = "".join(
            f'    <mesh name="{name}" file="{file}" />\n' for name, file in meshes_out
        )
        old = f'    <mesh name="{side}_rubber_hand" file="{side}_rubber_hand.STL" />\n'
        if text.count(old) != 1:
            raise SystemExit(f"{side}: rubber hand mesh asset not found")
        text = text.replace(old, asset_lines)

        # -- mimics
        for j in joints_out:
            m = j.find("mimic")
            if m is not None:
                equalities.append(
                    f'    <joint name="{j.get("name")}_follows" joint1="{j.get("name")}" '
                    f'joint2="{m.get("joint")}" polycoef="{num(float(m.get("offset") or 0))} '
                    f'{num(float(m.get("multiplier")))} 0 0 0" />\n'
                )

    eq = (
        "  <!-- The RH56E2's passive finger segments, from the URDF's <mimic> tags: the second\n"
        "       joint of each finger follows the first, and the thumb's last two follow its second. -->\n"
        "  <equality>\n" + "".join(equalities) + "  </equality>\n\n"
        "  <!-- Hand bodies bolted to the flange vs the wrist links they sit against. Hand-vs-hand\n"
        "       contact is already off through rh56e2_collision's bitmask. -->\n"
        "  <contact>\n" + "".join(excludes) + "  </contact>\n\n"
    )
    anchor = "  <actuator>\n"
    if text.count(anchor) != 1:
        raise SystemExit("could not find the <actuator> section")
    text = text.replace(anchor, eq + anchor)
    return text, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--urdf", required=True, type=pathlib.Path)
    parser.add_argument("--mjcf", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    urdf = Urdf(args.urdf)
    before = args.mjcf.read_text()
    after, stats = build(urdf, before)

    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/unitree_robots/g1/g1_29dof.xml",
            tofile="b/unitree_robots/g1/g1_29dof.xml",
            n=3,
        )
    )
    header = PATCH_HEADER.format(
        bodies=stats["bodies"],
        joints=stats["joints"],
        driven=stats["driven"],
        mimic=stats["mimic"],
        driven2=2 * stats["driven"],
        nu=29 + 2 * stats["driven"],
        mass=SIM_HAND_MASS_KG,
        real=stats["real_mass"],
    )
    args.out.write_text(header + "\n" + diff)
    print(
        f"wrote {args.out}: {stats['bodies']} bodies, {stats['joints']} joints "
        f"({stats['driven']} driven, {stats['mimic']} mimic) per hand, "
        f"real hand mass {stats['real_mass']:.3f} kg scaled to {SIM_HAND_MASS_KG:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
