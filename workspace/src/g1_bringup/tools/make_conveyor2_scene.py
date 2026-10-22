#!/usr/bin/env python3
"""Generate g1_conveyor2_scene.xml: the conveyor world with a stand-in part.

world:=conveyor2 is g1_conveyor_scene.xml exactly -- tray, conveyor, pallet, robot, everything --
except that every BSH panel (part_tray_1..4 and part_drop_test) is replaced by an upright red
drink can. It exists to build and exercise the pick-and-place system (tracking, Pick, Place into
a nest) with a part the hand can certainly hold, while the real panel's grasp waits for the real
hand. The bodies keep their names, so everything that looks them up still finds them.

The can's shape is g1_perception/models/coke_can.stl -- the tracker's model of it too, so the
depth tracker and the simulator agree on the shape exactly. It is inlined into the scene (a
<mesh> with vertex/face data), so the scene needs no mesh file staged next to the robot's.
MuJoCo collides with its convex hull, which for a can is the can.

The can, in its body frame (axis along z, top up):
  66 mm across, 128.7 mm tall (a Coca-Cola can model scaled to a real can's 66 mm diameter),
  0.35 kg (a full can), centre of mass at the middle of its height
  origin on the axis, 52.3 mm above the bottom -- NOT at the middle

The origin's height is chosen so that a can standing on the pallet plate (pallet z +0.01) at a
nest's centre -- between that nest's end cradles, which are 0.51 m apart -- has its origin at
pallet z 0.0623, inside the nest site (half height 8 mm) the simulator's conveyor handler
watches. A placed can therefore fills the nest and the pallet indexes, exactly as for a seated
panel. The grasps are published by the tracker, in this frame, so nothing assumes the origin
is in the middle.

    make_conveyor2_scene.py                      # default paths
    make_conveyor2_scene.py --from-source X.stl  # first rebuild coke_can.stl from a raw model

--from-source takes a can model in any unit and orientation-as-exported (long axis = the
model's y, top at +y, as Coca-Cola.stl is), keeps its outer surface only (the model is a 2 mm
shell: its inner wall would be "visible" surface to the tracker that the camera never sees),
scales it to DIAMETER, moves the origin as above and decimates it to FACES triangles (needs
trimesh and fast-simplification).
"""

import argparse
import pathlib
import re
import struct

import numpy as np

DIAMETER = 0.066
ORIGIN_ABOVE_BOTTOM = 0.0523
FACES = 4000
MASS = 0.35
RGBA = "0.72 0.02 0.04 1"

# Where the four tray parts stand (x as the panels in g1_conveyor_scene.xml, between the tray's
# low stop bars; on the tray floor, top z 0.71) and where the drop-test part starts (above the
# load station's nest, its bottom 10 mm above the tallest cradle wall, z 0.842, so it falls onto
# the plate between the cradles when the simulation starts).
TRAY_FLOOR = 0.71
TRAY_X = {1: 1.605, 2: 1.755, 3: 1.905, 4: 2.055}
TRAY_Y = -0.80
DROP_POS = (0.40, 0.60, 0.842 + 0.010 + ORIGIN_ABOVE_BOTTOM)


def _fmt(v):
    return " ".join(f"{x:.5g}" for x in v)


def read_stl(path):
    """Binary STL -> (vertices, faces), vertices merged."""
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    tri = np.frombuffer(data[84 : 84 + 50 * n], dtype=np.dtype("(3,)<f4, (3,3)<f4, <u2"))["f1"]
    verts, faces = np.unique(tri.reshape(-1, 3).round(7), axis=0, return_inverse=True)
    return verts.astype(float), faces.reshape(-1, 3)


def write_stl(path, verts, faces, name):
    with open(path, "wb") as fh:
        fh.write(name.encode().ljust(80, b" ")[:80])
        fh.write(struct.pack("<I", len(faces)))
        for f in faces:
            a, b, c = verts[f]
            n = np.cross(b - a, c - a)
            n = n / (np.linalg.norm(n) or 1.0)
            fh.write(struct.pack("<12fH", *n, *a, *b, *c, 0))


def from_source(src, dst):
    import trimesh

    m = trimesh.load(src)
    lo, hi = m.bounds
    s = DIAMETER / (hi[0] - lo[0])
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    v = m.vertices
    # model y (the can's axis, top at +y) -> body z; model (x, z) -> body (x, -y): right-handed
    body = np.c_[v[:, 0] - cx, -(v[:, 2] - cz), v[:, 1] - lo[1] - ORIGIN_ABOVE_BOTTOM / s] * s
    mesh = trimesh.Trimesh(body, m.faces, process=True)
    outer = max(mesh.split(only_watertight=False), key=lambda c: c.volume)
    outer = outer.simplify_quadric_decimation(face_count=FACES)
    write_stl(dst, outer.vertices, outer.faces, "coke_can (make_conveyor2_scene.py)")


def can_inertial(verts):
    zb, zt = verts[:, 2].min(), verts[:, 2].max()
    r, h = DIAMETER / 2, zt - zb
    ixx = MASS * (3 * r * r + h * h) / 12
    izz = MASS * r * r / 2
    return (0.0, 0.0, (zb + zt) / 2), (ixx, ixx, izz)


def body_xml(name, pos, com, inertia):
    return (
        f'    <body name="{name}" pos="{_fmt(pos)}">\n'
        f'      <joint type="free" name="{name}_joint" damping="0.2"/>\n'
        f'      <inertial pos="{_fmt(com)}" mass="{MASS:.4g}" diaginertia="{_fmt(inertia)}"/>\n'
        f'      <geom name="{name}_can" group="2" type="mesh" mesh="coke_can" '
        f'material="mat_coke_can" contype="1" conaffinity="1" friction="1 0.01 0.0001"/>\n'
        "    </body>"
    )


def main():
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--src", type=pathlib.Path, default=here.parent / "mjcf" / "g1_conveyor_scene.xml"
    )
    p.add_argument(
        "--dst", type=pathlib.Path, default=here.parent / "mjcf" / "g1_conveyor2_scene.xml"
    )
    p.add_argument(
        "--stl",
        type=pathlib.Path,
        default=here.parents[1] / "g1_perception" / "models" / "coke_can.stl",
    )
    p.add_argument("--from-source", type=pathlib.Path)
    a = p.parse_args()
    if a.from_source:
        from_source(a.from_source, a.stl)
    verts, faces = read_stl(a.stl)
    com, inertia = can_inertial(verts)

    s = a.src.read_text()
    bodies = {
        f"part_tray_{k}": (x, TRAY_Y, TRAY_FLOOR + ORIGIN_ABOVE_BOTTOM) for k, x in TRAY_X.items()
    }
    bodies["part_drop_test"] = DROP_POS
    for name, pos in bodies.items():
        m = re.search(rf'    <body name="{name}"[^>]*>.*?\n    </body>', s, re.S)
        if not m:
            raise SystemExit(f"{a.src}: no body {name}")
        s = s[: m.start()] + body_xml(name, pos, com, inertia) + s[m.end() :]
    asset = (
        "    <!-- conveyor2's part: a red drink can (make_conveyor2_scene.py, from\n"
        "         g1_perception/models/coke_can.stl). -->\n"
        f'    <material name="mat_coke_can" rgba="{RGBA}" specular="0.6" shininess="0.6"/>\n'
        f'    <mesh name="coke_can" vertex="{" ".join(f"{x:.5f}" for x in verts.ravel())}"\n'
        f'      face="{" ".join(str(int(i)) for i in faces.ravel())}"/>\n'
        "  </asset>"
    )
    s = s.replace("  </asset>", asset, 1)
    note = (
        "<!-- GENERATED by g1_bringup/tools/make_conveyor2_scene.py from g1_conveyor_scene.xml:\n"
        "     the same world with every BSH panel replaced by a red drink can (see that tool).\n"
        "     Do not edit by hand; edit g1_conveyor_scene.xml or the tool and regenerate. -->\n"
    )
    s = s.replace('<mujoco model="g1_conveyor_scene">', '<mujoco model="g1_conveyor2_scene">', 1)
    s = re.sub(r"(<mujoco[^>]*>\n)", r"\1" + note, s, count=1)
    a.dst.write_text(s)
    print(f"wrote {a.dst}" + (f" and {a.stl}" if a.from_source else ""))


if __name__ == "__main__":
    main()
