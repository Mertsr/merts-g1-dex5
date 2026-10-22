#!/usr/bin/env python3
"""Why is the octomap empty, or missing the table? One command, one verdict per link of the chain.

    ros2 run g1_manipulation octomap_check.py
    ros2 run g1_manipulation octomap_check.py --seconds 5 --table-z -0.08

Run it with the robot standing at the table (after go_to_stance.py), everything launched as for
a pick. It walks the chain the octomap is built from, in order, and stops pointing at the first
broken link:

  1. the clouds   -- /livox/lidar and /g1_depth_cloud/points: arriving? how many points, in
                     which frame, how old?
  2. TF           -- can each cloud's frame be put into the pelvis (the octomap's frame)?
  3. move_group   -- did it load the sensors (its `sensors` parameter), and do its self-filtered
                     clouds (/filtered_cloud, /filtered_depth_cloud) come out? Those are what
                     actually goes into the map, after the robot's own body is cut out.
  4. the map      -- /get_planning_scene's octomap: how many occupied voxels, and how many of
                     them form a horizontal layer in front of the robot at table height.
  5. RViz         -- /monitored_planning_scene, which the MotionPlanning display draws from:
                     does it carry the octomap at all?

Nothing is changed; it only listens and asks.
"""

import argparse
import struct
import sys
import time

import rclpy
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from rcl_interfaces.srv import GetParameters
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener

CLOUDS = {
    "/livox/lidar": "the Mid360 LiDAR",
    "/g1_depth_cloud/points": "the head depth camera (g1_depth_cloud)",
}
FILTERED = {
    "/filtered_cloud": "LiDAR after move_group's self-filter",
    "/filtered_depth_cloud": "depth camera after move_group's self-filter",
}
PLANNING_FRAME = "pelvis"


def occupied_leaves(data, resolution, depth=16):
    """Centres and sizes of the occupied leaves of a binary octomap (octomap_msgs, binary=True).

    OcTree::readBinaryData: depth-first, each node is two bytes of 2-bit child codes (children
    0-3 then 4-7): 00 unknown, 01 free leaf, 10 occupied leaf, 11 has children. A child's index
    bits are x (1), y (2), z (4). Keys run 0..2^16 with the origin at 2^15.
    """
    out = []
    pos = 0
    # (key of the node's min corner, depth)
    stack_root = (0, 0, 0)
    size_keys = 1 << depth

    def read(kx, ky, kz, d):
        nonlocal pos
        if pos + 2 > len(data):
            raise ValueError("truncated octomap")
        b0, b1 = data[pos], data[pos + 1]
        pos += 2
        codes = [(b0 >> (2 * i)) & 3 for i in range(4)] + [(b1 >> (2 * i)) & 3 for i in range(4)]
        half = size_keys >> (d + 1)
        children = []
        for i, c in enumerate(codes):
            cx = kx + (half if i & 1 else 0)
            cy = ky + (half if i & 2 else 0)
            cz = kz + (half if i & 4 else 0)
            if c == 2:
                out.append((cx, cy, cz, half))
            elif c == 3:
                children.append((cx, cy, cz))
        for cx, cy, cz in children:
            read(cx, cy, cz, d + 1)

    if data:
        read(*stack_root, 0)
    return _to_metres(out, resolution, depth)


def occupied_leaves_full(data, resolution, depth=16):
    """The same from a full octomap (binary=False, what /get_planning_scene sends).

    OcTree::readData: depth-first, each node is its log-odds as a float32, then one byte whose
    bit i says child i exists. A node without children is a leaf; occupied when its log-odds is
    above 0 (probability 0.5, OcTree's default threshold).
    """
    out = []
    pos = 0
    size_keys = 1 << depth

    def read(kx, ky, kz, d):
        nonlocal pos
        if pos + 5 > len(data):
            raise ValueError("truncated octomap")
        (logodds,) = struct.unpack_from("<f", data, pos)
        children = data[pos + 4]
        pos += 5
        n = size_keys >> d
        if children == 0:
            if logodds > 0.0:
                out.append((kx, ky, kz, n))
            return
        half = n >> 1
        for i in range(8):
            if children & (1 << i):
                read(kx + (half if i & 1 else 0), ky + (half if i & 2 else 0),
                     kz + (half if i & 4 else 0), d + 1)

    if data:
        read(0, 0, 0, 0)
    return _to_metres(out, resolution, depth)


def _to_metres(out, resolution, depth):
    origin = 1 << (depth - 1)
    leaves = []
    for kx, ky, kz, n in out:
        # A pruned leaf of n x n x n voxels; its centre in metres.
        c = [((k + n / 2.0) - origin) * resolution for k in (kx, ky, kz)]
        leaves.append((c[0], c[1], c[2], n * resolution))
    return leaves


class Check(Node):
    def __init__(self):
        super().__init__("octomap_check")
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.stats = {}
        for topic in list(CLOUDS) + list(FILTERED):
            self.stats[topic] = {"n": 0, "last": None}
            self.create_subscription(
                PointCloud2, topic, lambda m, t=topic: self.on_cloud(t, m), qos_profile_sensor_data
            )
        self.scene_msgs = []
        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(PlanningScene, "/monitored_planning_scene",
                                 self.scene_msgs.append, latched)

    def on_cloud(self, topic, msg):
        s = self.stats[topic]
        s["n"] += 1
        s["last"] = msg
        s["recv"] = time.time()

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def call(self, client, request, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        end = time.time() + timeout
        while not future.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return future.result() if future.done() else None


def say(ok, text):
    print(("  OK    " if ok else "  FAIL  ") + text)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=4.0, help="how long to listen (default 4)")
    ap.add_argument("--table-z", type=float, default=None,
                    help="the table top in the pelvis frame, if known (conveyor_flat: about -0.08)")
    args = ap.parse_args()

    rclpy.init()
    node = Check()
    print(f"listening for {args.seconds:.0f} s ...")
    node.spin_for(args.seconds)
    first_fail = None

    print("\n1. clouds")
    for topic, what in CLOUDS.items():
        s = node.stats[topic]
        if s["n"] == 0:
            say(False, f"{topic} ({what}): nothing arrived")
            continue
        m = s["last"]
        npts = m.width * m.height
        age = node.get_clock().now() - Time.from_msg(m.header.stamp)
        say(npts > 0, f"{topic}: {s['n'] / args.seconds:.1f} Hz, {npts} points, frame "
                      f"'{m.header.frame_id}', stamp {age.nanoseconds / 1e9:+.2f} s old")
        if abs(age.nanoseconds / 1e9) > 1.0:
            say(False, f"{topic}: stamps are {age.nanoseconds / 1e9:.1f} s off this clock -- "
                       "the octomap's TF filter drops clouds it cannot place in time")
            first_fail = first_fail or f"{topic} stamps are off the wall clock"

    if all(node.stats[t]["n"] == 0 for t in CLOUDS):
        first_fail = "no cloud is publishing at all (sensors:=true? the simulator's sensor relay?)"

    print("\n2. TF into the planning frame")
    for topic in CLOUDS:
        m = node.stats[topic]["last"]
        if m is None:
            continue
        try:
            node.tf_buffer.lookup_transform(PLANNING_FRAME, m.header.frame_id, Time.from_msg(m.header.stamp),
                                            timeout=Duration(seconds=0.5))
            say(True, f"{m.header.frame_id} -> {PLANNING_FRAME} at the cloud's stamp")
        except Exception as e:  # noqa: BLE001 -- any TF failure is the answer here
            try:
                node.tf_buffer.lookup_transform(PLANNING_FRAME, m.header.frame_id, Time())
                say(False, f"{m.header.frame_id} -> {PLANNING_FRAME} exists, but not at the cloud's "
                           f"stamp ({type(e).__name__}) -- the clouds are dropped by the TF filter")
            except Exception as e2:  # noqa: BLE001
                say(False, f"{m.header.frame_id} -> {PLANNING_FRAME}: no transform at all ({e2})")
            first_fail = first_fail or f"no TF from {m.header.frame_id} to {PLANNING_FRAME}"

    print("\n3. move_group")
    params = node.create_client(GetParameters, "/move_group/get_parameters")
    req = GetParameters.Request()
    req.names = ["sensors", "octomap_resolution"]
    res = node.call(params, req)
    resolution = 0.025
    if res is None:
        say(False, "/move_group/get_parameters did not answer -- is move_group running?")
        first_fail = first_fail or "move_group is not answering"
    else:
        sensors = list(res.values[0].string_array_value)
        if res.values[1].type == 3:
            resolution = res.values[1].double_value
        say(bool(sensors), f"sensors loaded: {sensors or 'NONE -- sensors_3d.yaml did not reach move_group'}; "
                           f"resolution {resolution} m")
        if not sensors:
            first_fail = first_fail or "move_group has no sensors configured"
    # Is move_group subscribed to the clouds at all, and can its subscription hear the publisher?
    # Nothing in the filtered clouds with the input publishing means one of: the updater plugin
    # never started (no subscription), a QoS pair that cannot match (a RELIABLE subscriber does
    # not hear a BEST_EFFORT publisher), or clouds that arrive and are dropped later.
    try:
        from ament_index_python.packages import get_package_share_directory
        import xml.etree.ElementTree as ET
        share = get_package_share_directory("moveit_ros_perception")
        version = ET.parse(share + "/package.xml").getroot().findtext("version")
        say(True, f"moveit_ros_perception {version} installed (the PointCloudOctomapUpdater plugin)")
    except Exception as e:  # noqa: BLE001
        say(False, f"moveit_ros_perception is not installed ({type(e).__name__}): move_group cannot load "
                   "the octomap updater -- sudo apt install ros-humble-moveit-ros-perception")
        first_fail = first_fail or "moveit_ros_perception (the octomap updater plugin) is missing"
    for topic in CLOUDS:
        subs = [i for i in node.get_subscriptions_info_by_topic(topic) if i.node_name == "move_group"]
        pubs = node.get_publishers_info_by_topic(topic)
        pub_rel = sorted({p.qos_profile.reliability.name for p in pubs})
        if not subs:
            say(False, f"{topic}: move_group is NOT subscribed (publishers: "
                       f"{[p.node_name for p in pubs]}) -- its updater for this sensor never started; "
                       "move_group's log at startup says why (search it for 'octomap' / 'updater')")
            first_fail = first_fail or f"move_group has no subscription on {topic}"
            continue
        sub_rel = subs[0].qos_profile.reliability.name
        mismatch = sub_rel == "RELIABLE" and "BEST_EFFORT" in pub_rel
        say(not mismatch, f"{topic}: move_group subscribed {sub_rel}, published {pub_rel} by "
                          f"{[p.node_name for p in pubs]}"
                          + (" -- a RELIABLE subscriber never hears a BEST_EFFORT publisher" if mismatch else ""))
        if mismatch:
            first_fail = first_fail or f"QoS mismatch on {topic}"
    for topic, what in FILTERED.items():
        s = node.stats[topic]
        if s["n"] == 0:
            source = list(CLOUDS)[list(FILTERED).index(topic)]
            say(False, f"{topic} ({what}): nothing -- move_group is not taking that cloud in")
            if node.stats[source]["n"] > 0:
                first_fail = first_fail or (
                    f"move_group is subscribed to {source}, which IS publishing, yet maps nothing: the "
                    "clouds are dropped inside move_group (its TF message filter, or the self-filter's "
                    "transform cache) -- its log says 'Transform error' or 'Transform cache'")
        else:
            m = s["last"]
            say(m.width * m.height > 0, f"{topic}: {s['n'] / args.seconds:.1f} Hz, {m.width * m.height} "
                                        "points left after the self-filter")

    print("\n4. the octomap")
    scene = node.create_client(GetPlanningScene, "/get_planning_scene")
    req = GetPlanningScene.Request()
    req.components.components = PlanningSceneComponents.OCTOMAP
    res = node.call(scene, req)
    if res is None:
        say(False, "/get_planning_scene did not answer")
        first_fail = first_fail or "/get_planning_scene is not answering"
    else:
        om = res.scene.world.octomap
        data = bytes((b + 256) % 256 for b in om.octomap.data)
        say(True, f"frame '{om.header.frame_id}', {om.octomap.id}, binary={om.octomap.binary}, "
                  f"{len(data)} bytes, resolution {om.octomap.resolution}")
        try:
            parse = occupied_leaves if om.octomap.binary else occupied_leaves_full
            leaves = parse(data, om.octomap.resolution or resolution)
        except Exception as e:  # noqa: BLE001
            leaves = []
            say(False, f"could not read it: {e}")
        n_vox = sum((s / (om.octomap.resolution or resolution)) ** 3 for *_, s in leaves)
        say(n_vox > 0, f"{int(n_vox)} occupied voxels")
        if n_vox == 0:
            first_fail = first_fail or "the octomap is empty"
        # In front of the robot, within arm's reach, below the pelvis: where the table is.
        front = [(x, y, z) for x, y, z, s in leaves if 0.2 < x < 1.0 and abs(y) < 0.7 and -0.4 < z < 0.2]
        if front:
            hist = {}
            for _, _, z in front:
                k = round(z, 4)  # voxel centres are already on the grid
                hist[k] = hist.get(k, 0) + 1
            top = sorted(hist.items(), key=lambda kv: -kv[1])[:4]
            print("        in front of the robot (pelvis frame, x 0.2-1.0 m, |y| < 0.7 m), voxels per height:")
            for z, n in sorted(top):
                print(f"          z {z:+.3f} m: {n}")
            layer_z, layer_n = top[0]
            flat = layer_n >= 40
            say(flat, f"densest layer at pelvis z {layer_z:+.3f} m ({layer_n} voxels)"
                      + ("" if flat else " -- too sparse to be a table"))
            if args.table_z is not None:
                say(abs(layer_z - args.table_z) < 0.04,
                    f"expected the table at z {args.table_z:+.3f}: off by {layer_z - args.table_z:+.3f} m")
            if not flat:
                first_fail = first_fail or "the table is not in the octomap"
        else:
            say(False, "nothing occupied in front of the robot at table height")
            first_fail = first_fail or "the table is not in the octomap"

    print("\n5. what RViz gets (/monitored_planning_scene)")
    with_map = [m for m in node.scene_msgs if len(m.world.octomap.octomap.data) > 0]
    if not node.scene_msgs:
        say(False, "no message in the listening window (it only publishes on change; "
                   "an empty octomap never changes)")
    else:
        say(bool(with_map), f"{len(node.scene_msgs)} messages, {len(with_map)} carrying the octomap")
    print("        RViz: MotionPlanning > Scene Geometry > Show Scene Geometry ON, Voxel Rendering "
          "'Occupied Voxels'; Fixed Frame must resolve to pelvis through TF")

    print()
    if first_fail:
        print(f"FIRST BROKEN LINK: {first_fail}")
    else:
        print("The chain is intact: the clouds arrive, move_group maps them and the table is in the octomap.")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if first_fail is None else 1


if __name__ == "__main__":
    sys.exit(main())
