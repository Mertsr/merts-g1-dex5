#!/usr/bin/env python3
"""Export the conveyor pallet -- plate and all its nests -- as one solid STL.

Built from make_conveyor_pallet.py's pallet_geometry(), the same box list the scene's pallet
block is generated from, so the part and the simulated pallet cannot drift apart: change a
dimension there, rerun both. Every box is unioned into a single watertight solid, so it opens
cleanly in CAD and slices for printing.

Writes two files:
  BSH_pallet_mm.stl   millimetres, z up, the plate's underside at z = 0, centred on x and y,
                      x along the conveyor's travel (nest 1 at +x). For making the part.
  BSH_pallet_m.stl    metres, in the simulation's own conveyor_pallet body frame (origin at the
                      plate's centre). Drop-in for MuJoCo or for checking against the scene.

    export_conveyor_pallet_stl.py --clearance-mm 2.0 --nests 4 --out-dir .

Needs numpy, trimesh and manifold3d (pip install trimesh manifold3d).
"""

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import trimesh


def load_generator():
    path = pathlib.Path(__file__).resolve().parent / "make_conveyor_pallet.py"
    spec = importlib.util.spec_from_file_location("make_conveyor_pallet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pallet_solid(geo):
    boxes = []
    for _name, half, pos in geo["boxes"]:
        b = trimesh.creation.box(extents=2 * np.asarray(half, dtype=float))
        b.apply_translation(pos)
        boxes.append(b)
    solid = trimesh.boolean.union(boxes, engine="manifold")
    if not solid.is_watertight or len(solid.split()) != 1:
        raise SystemExit("the pallet did not union into one watertight solid")
    return solid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clearance-mm", type=float, default=2.0)
    parser.add_argument("--nests", type=int, default=4)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("."))
    args = parser.parse_args(argv)

    g = load_generator()
    geo = g.pallet_geometry(args.clearance_mm / 1000.0, args.nests)
    solid = pallet_solid(geo)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    solid.export(args.out_dir / "BSH_pallet_m.stl")  # simulation's pallet body frame, metres

    mm = solid.copy()
    mm.apply_translation((0.0, 0.0, g.PLATE_HALF_T))  # plate underside to z = 0
    mm.apply_scale(1000.0)
    mm.export(args.out_dir / "BSH_pallet_mm.stl")

    ext = mm.extents
    print(f"wrote {args.out_dir}/BSH_pallet_mm.stl and BSH_pallet_m.stl: one solid, {args.nests} "
          f"nests at {geo['pitch'] * 1000:.1f} mm pitch, clearance {args.clearance_mm:g} mm, "
          f"{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
