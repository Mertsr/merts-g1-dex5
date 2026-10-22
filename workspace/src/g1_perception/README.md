# g1_perception

Model-based 6D pose tracking of the line's known parts (the pallet and the BSH panel) from the
head depth camera, at camera rate. The same code runs in the simulator and on the robot.

```
/camera/aligned_depth_to_color/image_raw ─┐
/camera/aligned_depth_to_color/camera_info ┼─> g1_object_tracker ──> /g1_object_tracker/detections
TF (odom -> camera)  ─────────────────────┘          │                 (Detection3DArray, camera frame)
                                                     ├──> TF tracked/<name> (under odom)
                                                     └──> /diagnostics
/g1_object_tracker/detections ──> g1_object_pose_source (object_source:=depth_tracker) ──> /objects
```

## Running it

```bash
# Simulator, tracker only (forces sensors:=true). Look at tracked/pallet and tracked/panel in RViz.
ros2 launch g1_bringup bringup.launch.py world:=conveyor object_tracker:=true rviz:=true

# Skills reading tracked poses instead of ground truth (the tracker starts automatically):
ros2 launch g1_bringup bringup.launch.py world:=conveyor moveit:=true manipulation:=true \
    object_source:=depth_tracker

# world:=conveyor2 is the same line with a red drink can (66 x 129 mm) instead of each BSH panel
# (g1_bringup/tools/make_conveyor2_scene.py); the tracker then loads
# config/object_tracker_conveyor2.yaml (models/coke_can.stl, objects can_1..4, symmetric_z, side
# grasps side_right_* / side_left_* taken in g1.srdf's side_open / side_closed) by itself:
ros2 launch g1_bringup bringup.launch.py world:=conveyor2 moveit:=true manipulation:=true \
    object_source:=depth_tracker

# world:=conveyor_flat is the line with the panels on a plain table: no tray rim, no stop bars, as
# on the real line (g1_bringup/tools/make_conveyor_flat_scene.py). The tracker then loads
# config/object_tracker_conveyor_flat.yaml: the panel config plus the scoop grasps
# (scoop_<hand>_<edge>, g1.srdf's scoop_open / scoop_closed), which slide the hand in under the
# panel's edge along the table:
ros2 launch g1_bringup bringup.launch.py world:=conveyor_flat moveit:=true manipulation:=true \
    object_source:=depth_tracker

# On its own, against any depth stream with TF:
ros2 launch g1_perception object_tracker.launch.py config:=/path/to/line.yaml
```

The simulator's camera now renders on its own thread at `camera_rate_hz` (30 Hz in
`g1_bringup/config/sim_sensors.yaml`) instead of once per 10 Hz LiDAR sweep; rebuild the patched
`unitree_mujoco` after pulling. Its log line `[grove_g1] camera: X ms/frame (budget 33.3)` shows
whether the machine keeps up.

## Watching it work

```bash
ros2 launch g1_bringup bringup.launch.py world:=conveyor object_tracker:=true object_tracker_rviz:=true
# or, beside a running tracker:
rviz2 -d $(ros2 pkg prefix g1_perception)/share/g1_perception/rviz/perception.rviz
# or just the picture:
ros2 run rqt_image_view rqt_image_view /g1_object_tracker/debug/image
```

Three views, computed only while something subscribes to them:

| Topic | Shows |
| --- | --- |
| `~/debug/image` | The depth image (near bright), and per object: the region it fitted; its model's camera-facing surface at the fitted pose, in the object's colour where the depth confirms it and **red** where the camera sees through it (what rejects a wrong pose); **magenta** fitted points inside the part's box that missed its surface (what lowers the inlier ratio); while searching, every pose being tried, as boxes. A status line per object at the top: `TRACK` / `REJECT` / `SEARCH`, inlier ratio, coverage, violation, points, time. |
| `~/debug/markers` | In RViz's 3D view: each model's mesh where it was fitted (red when this frame's fit was rejected), a label with its scores, and the search boxes. |
| `~/debug/points` | The points each fit used, in 3D: object colour on the model, magenta in its box but off it, grey outside it. |

Offline, `icp_bench` writes the same image: `DEBUG_DIR=/tmp/view DEBUG_EVERY=5 icp_bench ...`
gives `/tmp/view/NNNNN.ppm`.

## How it works

Per frame, per object, in the configured order:

1. **Predict.** The object's pose is kept in `odom` with a constant-velocity model and
   re-expressed through the camera's TF at the image stamp. The head moves with every step;
   carried in the camera frame, that motion would look like the part jumping.
2. **Cut out.** Only the model's projected bounding box is back-projected (about 3000 points),
   and every point another tracked object explains is removed. This is why the pallet comes
   first: a panel seated with 2 mm clearance otherwise fits itself to the nest's walls.
3. **ICP.** Point-to-plane, Huber-weighted, with correspondences from a nearest-neighbour grid
   precomputed around the model (one array read per point), back-facing surfaces rejected (with
   a twin lookup across thin walls), and a 12 → 5 mm correspondence gate.
4. **Verify.** The model's camera-facing surface is compared with the depth image: the share the
   depth confirms (coverage), and the share where the camera sees *past* it (violation, meaning
   the part cannot be there). Occlusion counts as neither, so a hand over the part does not fail
   the check. Five bad frames in a row and the object is searched for again.

**Searching** runs on a worker thread, so one object's search never stalls another's tracking.
It tries the configured **priors** first: the pallet's station on the line, swept along its
travel; the four nests relative to the tracked pallet; optionally the hand. Each prior gets a
wide ICP pass, then a tight one, then verification. After `global_after_s` without a match,
FPFH + RANSAC over the whole frame runs as well.

## Results

These come from `tools/make_icp_dataset.py` and `icp_bench`. They are head-camera renders of
the conveyor scene at 848x480 with RealSense-like noise: σ = 0.6 mm + 2.5 mm·z², 1 mm
quantisation, 2 % dropouts and invalid depth edges. Both parts are tracked together, on a 2-core
sandbox.

| Sequence | Object | Valid frames | Mean error | Max error | Time per object |
| --- | --- | --- | --- | --- | --- |
| static | pallet | 100 % | 0.36 mm / 0.04° | 0.75 mm / 0.09° | 1.5 ms |
| static | panel in nest 1 | 100 % | 0.36 mm / 0.18° | 0.89 mm / 0.50° | 1.0 ms |
| head sway + pallet indexing 150 mm | pallet | 100 % | 0.29 mm / 0.04° | 0.82 mm / 0.15° | 1.9 ms |
| head sway + pallet indexing 150 mm | panel | 100 % | 0.61 mm / 0.24° | 1.69 mm / 0.64° | 1.0 ms |
| head sway + panel carried and turned | panel | 100 % | 0.56 mm / 0.23° | 2.20 mm / 1.42° | 0.9 ms |

All objects together take 2.4 to 2.8 ms per frame on average, at most 5.8 ms, which is over
350 fps of tracking budget against a 30 fps camera. Lock-on from the priors takes 3 to 5 frames.
Model preparation takes about 3 s for the pallet and 1 s for the panel at start-up.

Reproduce:

```bash
MUJOCO_GL=egl python3 tools/make_icp_dataset.py <scene.xml> /tmp/icpdata   # needs mujoco, numpy
icp_bench /tmp/icpdata/sway_index pallet=models/bsh_pallet.stl panel=models/bsh_panel.stl
```

## Grasps and slots

Per object, in its output frame (`config/object_tracker_conveyor.yaml`):

- `grasps: ["<name> x y z roll pitch yaw"]`: where a hand takes the part.
- `slots: ["<name> x y z roll pitch yaw"]`: where another part sits on this one, such as the
  pallet's nests.

The tracker publishes each one, carried by the tracked pose, in three ways:

- as an extra hypothesis on the detection: `panel/grasp/edge_high`, `pallet/slot/nest_2`;
- as a TF frame: `tracked/panel/grasp/edge_high`;
- in RViz (`~/debug/markers`):
  - a grasp is drawn as a gripper: approach line yellow, fingers red, thumb side blue, palm white,
    a cyan cross where the hand closes;
  - a slot is drawn as axes.

`g1_manipulation`'s Pick and Place use them; see its README. The panel's `edge_high` grasp is a
first guess from the CAD. Check it against the hand in the simulator before relying on it.

## Configuration

See `config/object_tracker_conveyor.yaml`. Poses are one string each:
`"<frame> x y z [roll pitch yaw]"` (metres, radians, RPY about fixed axes), where `<frame>` is a
TF frame or another tracked object's name. `prior_sweep: "x 0.0 0.75 0.025"` tries a prior
every 25 mm along an axis.

Per object, besides the model, priors and offsets:

| Key | What |
| --- | --- |
| `instances: N` | Track up to N of this part at once, as `<name>_1` .. `<name>_N`, sharing one loaded model. Pick with `object_id: <name>` takes the nearest. |
| `region: "<frame> xmin xmax ymin ymax zmin zmax [max_angle_deg [roll pitch yaw]]"` | Where the object can be. A fit outside it is dropped and the object searched for again, never reported. The pallet's keeps it on its line. |
| `instance_separation` | How close two instances' poses may be (default 0.05 m). |
| `symmetric_z: true` | The part is round about its own z (a can). Its yaw is reported facing the world frame's +x instead of whatever the fit gave, so its grasps and slots stay put, and the detection carries `<name>/symmetric/z` so a place may turn it freely about that axis. |

**Swapping in scanned models.** Set `<object>.model` to the PLY, with `model_scale: 0.001` if it
is in millimetres. PLY with faces is best, because normals then come from the faces and
back-face rejection works. A point-only PLY works too. The priors and `output_offset` are
written for the model's own frame: if the scan's origin differs from the STL's, re-measure them.
The easiest way is to let the tracker lock once from a hand-placed prior, then read the
`tracked/<name>` TF.

**On the robot:**

- The pallet's station prior is a position in `odom`, which is only right while odom starts
  where the simulator's does. Write it in `map` and run localization.
- Real D435i depth is 16UC1 in millimetres; `depth_scale: 0.001` handles it.
- For a panel in the hand, add a prior on the palm frame with the grasp offset.

## Limitations

- **FPFH rarely finds these parts on its own.** The pallet repeats every 150.6 mm and both
  parts are mostly flat. In tests it never produced a wrong lock, because verification rejects
  its bad answers. It also never found a part without a prior. Configure priors for every place
  a part can be.
- **ICP cannot see sliding along a direction no visible face resists.** A part whose faces
  across that direction all face away from the camera can settle up to about a centimetre off,
  and still pass verification. On the line's views this did not happen, but a camera looking
  straight along a part's length would show it.
- **Instances share priors, not tracks.** `instances: 4` reports `panel_1` .. `panel_4`. An
  instance does not search a prior another one already holds, and a fit within
  `instance_separation` (5 cm) of another instance's part is dropped. With more parts in view than
  instances, the rest are not reported.
- **Mirror-symmetric placements are told apart only by asymmetric features.** On the panel these
  are its windows and clips; a panel flipped 180° in a nest is a separate prior.

## Files

| Path | What |
| --- | --- |
| `include/g1_perception/object_model.hpp` | Model loading (STL / PLY), sampling, NN grid (jump flooding), FPFH features. |
| `include/g1_perception/icp_tracker.hpp` | ICP, scoring, verification, global search, `ObjectTracker`, `ConstantVelocity`. |
| `include/g1_perception/debug_render.hpp` | The debug image: fits, verdicts and status drawn on the depth. |
| `rviz/perception.rviz` | RViz on the debug views. |
| `include/g1_perception/multi_tracker.hpp` | Several objects, order, explained points, priors through frames. What the node runs. |
| `src/object_tracker_node.cpp` | The ROS node: messages and TF only. |
| `tools/icp_bench.cpp` | Offline replay and grading against ground truth. |
| `tools/make_icp_dataset.py` | Renders test sequences from the MuJoCo scene. |
| `test/test_tracker.cpp` | Synthetic-scene tests of the core. |
