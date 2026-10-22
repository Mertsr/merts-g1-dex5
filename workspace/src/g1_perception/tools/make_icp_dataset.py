#!/usr/bin/env python3
"""Renders head-camera depth sequences of the conveyor scene for icp_bench.

Every sequence is posed kinematically: the robot stays at its keyframe, and the pallet and panel
are moved by writing qpos. Ground truth is exact. Physics is used only once, to let the drop-test
panel settle into nest 1 so the pallet-to-panel offset is the real seated one.

The depth is MuJoCo's metric depth with RealSense-like corruption: axial noise growing with z^2,
1 mm quantisation, random dropouts and invalid pixels along depth edges.

Output, one directory per sequence (format read by icp_bench):
  meta.txt        width height fx fy cx cy frames
  cam.txt         world_T_camera_optical per frame (3x4, row-major)
  gt_pallet.txt   world_T_pallet model (frame of cad/BSH_pallet_m.stl)
  gt_panel.txt    world_T_panel model (frame of the BSH_panel.stl file)
  depth/NNNNN.bin float32 metres, 0 = no reading
  priors.txt      what the robot would know beforehand, as the ROS node is configured with it:
                  the pallet's station pose at index 0 and its travel, the nests on the pallet,
                  and (carried sequence) the hand holding the panel
  pose_hand.txt   world_T_hand per frame (carried sequence): the panel's true pose with a fixed
                  grasp error, standing in for the robot's palm frame

  make_icp_dataset.py <scene.xml> <out_dir> [--frames-scale 1.0] [--seed 1]
"""
import argparse
import os

import mujoco
import numpy as np

W, H, FOVY = 848, 480, 58.0
FPS = 30.0
PANEL_REFPOS = np.array([0.0, 0.001, -0.046])  # part_mesh refpos in the scene


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def mat_to_quat(r):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(r, dtype=float).reshape(9))
    return q


def iso(r, t):
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T


def look_at(eye, target, up=(0, 0, 1)):
    """world_T_optical (z forward, x right, y down) for a camera at eye looking at target."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return iso(np.column_stack([x, y, z]), eye)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


class Scene:
    def __init__(self, path):
        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        self.cam = self.m.camera("d435i").id
        self.pallet_q = self.m.joint("conveyor_pallet_slide").qposadr[0]
        self.panel_q = self.m.joint("part_drop_test_joint").qposadr[0]
        self.panel_body = self.m.body("part_drop_test").id
        self.pallet_body = self.m.body("conveyor_pallet").id
        self.r = mujoco.Renderer(self.m, H, W)
        self.r.enable_depth_rendering()
        self.settle()

    def settle(self):
        """Drops the test panel into nest 1 with physics, the robot held at its keyframe."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        robot = slice(0, 7 + 29)
        robot_v = slice(0, 6 + 29)
        q0 = d.qpos[robot].copy()
        for _ in range(int(3.0 / m.opt.timestep)):
            d.qpos[robot] = q0
            d.qvel[robot_v] = 0
            d.qpos[self.pallet_q] = 0
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        self.qpos0 = d.qpos.copy()
        self.qpos0[robot] = q0
        w_pal = self.body_pose(self.pallet_body)
        w_pan = self.body_pose(self.panel_body)
        self.pal_T_panel = np.linalg.inv(w_pal) @ w_pan
        print("panel settled at", np.round(w_pan[:3, 3], 4))

    def body_pose(self, b):
        return iso(self.d.xmat[b].reshape(3, 3), self.d.xpos[b].copy())

    def pose(self, slide, world_T_panel_body=None):
        d = self.d
        d.qpos[:] = self.qpos0
        d.qvel[:] = 0
        d.qpos[self.pallet_q] = slide
        mujoco.mj_kinematics(self.m, d)
        if world_T_panel_body is None:  # riding in nest 1
            world_T_panel_body = self.body_pose(self.pallet_body) @ self.pal_T_panel
        d.qpos[self.panel_q:self.panel_q + 3] = world_T_panel_body[:3, 3]
        d.qpos[self.panel_q + 3:self.panel_q + 7] = mat_to_quat(world_T_panel_body[:3, :3])
        mujoco.mj_forward(self.m, d)

    def render(self, world_T_opt):
        # MuJoCo cameras look down -z with y up: optical = mujoco_cam * Rx(pi).
        w_mj = world_T_opt[:3, :3] @ rot_x(np.pi)
        self.m.cam_pos[self.cam] = world_T_opt[:3, 3]
        self.m.cam_quat[self.cam] = mat_to_quat(w_mj)
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera=self.cam)
        return self.r.render().astype(np.float32)

    def gt(self):
        w_pal = self.body_pose(self.pallet_body)
        w_pan_file = self.body_pose(self.panel_body) @ iso(np.eye(3), -PANEL_REFPOS)
        return w_pal, w_pan_file


def corrupt(z, rng):
    z = z.copy()
    z[(z <= 0.1) | (z > 4.0)] = 0
    sigma = 0.0006 + 0.0025 * z * z
    z += rng.normal(0, 1, z.shape).astype(np.float32) * sigma
    z = np.round(z * 1000) / 1000
    gy, gx = np.gradient(z)
    z[np.hypot(gx, gy) > 0.03] = 0  # flying pixels at depth edges read as nothing
    z[rng.random(z.shape) < 0.02] = 0
    return z.astype(np.float32)


def write_priors(scene, path, hand):
    """The station pose at index 0, the travel to try along, the nests, the hand."""
    row = lambda T: " ".join(f"{v:.7f}" for v in T[:3, :].reshape(-1))
    scene.pose(0.0)
    w_pal0, _ = scene.gt()
    pal_T_panel = scene.pal_T_panel @ iso(np.eye(3), -PANEL_REFPOS)
    pitch = 0.1506
    with open(os.path.join(path, "priors.txt"), "w") as f:
        f.write(f"prior pallet world {row(w_pal0)}\n")
        for x in np.arange(0.0, 0.75 + 1e-9, 0.025):
            f.write(f"offset pallet {row(iso(np.eye(3), [x, 0, 0]))}\n")
        for k in range(4):
            f.write(f"prior panel pallet {row(iso(np.eye(3), [-pitch * k, 0, 0]) @ pal_T_panel)}\n")
        if hand:
            f.write(f"prior panel hand {row(np.eye(4))}\n")


GRASP_ERROR = iso(rot_z(np.radians(4)) @ rot_x(np.radians(-3)), [0.008, -0.006, 0.005])


def write_seq(scene, out, name, frames, cam_at, panel_at, slide_at, rng):
    path = os.path.join(out, name)
    os.makedirs(os.path.join(path, "depth"), exist_ok=True)
    fy = (H / 2) / np.tan(np.radians(FOVY / 2))
    with open(os.path.join(path, "meta.txt"), "w") as mf:
        mf.write(f"{W} {H} {fy:.6f} {fy:.6f} {(W - 1) / 2:.3f} {(H - 1) / 2:.3f} {frames}\n")
    fc = open(os.path.join(path, "cam.txt"), "w")
    fp = open(os.path.join(path, "gt_pallet.txt"), "w")
    fn = open(os.path.join(path, "gt_panel.txt"), "w")
    fh_hand = open(os.path.join(path, "pose_hand.txt"), "w") if panel_at(0.0) is not None else None
    row = lambda T: " ".join(f"{v:.7f}" for v in T[:3, :].reshape(-1)) + "\n"
    for i in range(frames):
        t = i / FPS
        scene.pose(slide_at(t), panel_at(t))
        cam = cam_at(t)
        depth = corrupt(scene.render(cam), rng)
        w_pal, w_pan = scene.gt()
        depth.tofile(os.path.join(path, "depth", f"{i:05d}.bin"))
        fc.write(row(cam))
        fp.write(row(w_pal))
        fn.write(row(w_pan))
        if fh_hand:
            fh_hand.write(row(w_pan @ GRASP_ERROR))
    for fh in (fc, fp, fn, fh_hand):
        if fh:
            fh.close()
    write_priors(scene, path, fh_hand is not None)
    print(f"{name}: {frames} frames -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    sc = Scene(a.scene)

    # Head camera of a robot standing beside the load station, looking down at the pallet.
    station = np.array([0.40, 0.60, 0.83])
    eye0 = np.array([0.40, 0.08, 1.25])
    still = lambda t: look_at(eye0, station)

    def sway(t):
        # Walking-in-place head motion: a few centimetres and degrees at gait frequency.
        e = eye0 + np.array([0.025 * np.sin(2 * np.pi * 1.2 * t), 0.01 * np.sin(2 * np.pi * 0.7 * t),
                             0.012 * np.sin(2 * np.pi * 2.4 * t)])
        T = look_at(e, station)
        T[:3, :3] = rot_z(np.radians(3) * np.sin(2 * np.pi * 0.8 * t)) @ T[:3, :3]
        return T

    pitch = 0.1506
    indexing = lambda t: pitch * min_jerk((t - 1.0) / 2.0)

    def carried(t):
        # A panel carried from the side into view, over the station, turning as it goes.
        s = min_jerk(t / 4.0)
        p = np.array([0.85 - 0.45 * s, 0.50, 1.00 - 0.07 * s])
        R0 = sc.body_pose(sc.pallet_body)[:3, :3] @ sc.pal_T_panel[:3, :3]
        return iso(rot_z(np.radians(35) * (1 - s)) @ rot_x(np.radians(15) * np.sin(np.pi * s)) @ R0, p)

    write_seq(sc, a.out, "static", 60, still, lambda t: None, lambda t: 0.0, rng)
    write_seq(sc, a.out, "sway_index", 150, sway, lambda t: None, indexing, rng)
    write_seq(sc, a.out, "carried", 120, sway, carried, lambda t: 0.0, rng)


if __name__ == "__main__":
    main()
