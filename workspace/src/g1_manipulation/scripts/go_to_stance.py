#!/usr/bin/env python3
"""Walk the G1 to a named stance: a fixed x, y and heading on the floor, the same every time.

    ros2 run g1_manipulation go_to_stance.py panel_1        # walk there and stop
    ros2 run g1_manipulation go_to_stance.py --list         # the stances it knows
    ros2 run g1_manipulation go_to_stance.py panel_1 --check   # only say how far off it is
    ros2 run g1_manipulation go_to_stance.py --save pallet  # remember where it stands now
    ros2 run g1_manipulation go_to_stance.py --x 1.12 --y -0.78 --yaw-deg 0   # an ad-hoc one

No perception: from a metre or two away the head camera does not see the panels, so the stance
is a pose in a fixed frame (odom by default; in the simulator with odometry:=sportmodestate,
odom is the MuJoCo world itself, so these are world coordinates). The robot's ground frame
(base_footprint) is driven there closed loop on TF.

STANCES. From the package's config/stances.yaml, then ~/.ros/g1_stances.yaml on top (where
--save writes, so a stance you found by hand survives a rebuild). Each is
`name: {x: .., y: .., yaw_deg: .., frame: odom}`.

panel_1 (conveyor_flat): the middle of the stances the two-hand scoop of the first panel works
from, measured over a grid in MuJoCo with the scoop's turned placements (see the package
README): base_footprint at (1.12, -0.78), facing +x, 28 cm back from the table's edge. Every
stance within about +-10 cm and +-10 degrees of it worked there, so the tolerances here
(2 cm, 3 degrees) are far inside what the pick needs.

HOW IT WALKS. Straight there -- there is no obstacle avoidance: start with a clear line to the
stance (in front of the table, not beside it). More than `--near` (1.0 m) away it turns to face
the stance and walks to it, steering; nearer, it moves forward, sideways and turns at once,
each proportional to its error, to the stance's own heading. Commands are held above the gait
shaper's dead zone (0.05 m/s, 0.10 rad/s) until inside tolerance and go to /cmd_vel_approach,
the gait shaper's priority input -- so it needs nav:=true. Done after 2 s inside tolerance.

Arms first: tuck them (ready_arms.py) before walking up to the table, so a hand held out does not
meet its edge.
"""

import argparse
import math
import os
import sys
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

USER_FILE = os.path.expanduser("~/.ros/g1_stances.yaml")


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def command(error, gain, floor, cap, tolerance):
    """Proportional, but at least `floor` (the gait's dead zone) while outside tolerance."""
    if abs(error) <= tolerance:
        return 0.0
    v = gain * error
    return math.copysign(min(max(abs(v), floor), cap), v)


def load_stances():
    stances = {}
    try:
        share = get_package_share_directory("g1_manipulation")
        with open(os.path.join(share, "config", "stances.yaml")) as f:
            stances.update((yaml.safe_load(f) or {}).get("stances", {}))
    except (OSError, LookupError) as e:
        print(f"no package stances: {e}", file=sys.stderr)
    if os.path.exists(USER_FILE):
        with open(USER_FILE) as f:
            stances.update((yaml.safe_load(f) or {}).get("stances", {}))
    return stances


class GoToStance(Node):
    def __init__(self, a):
        super().__init__("go_to_stance")
        self.a = a
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.cmd = self.create_publisher(Twist, "/cmd_vel_approach", 10)

    def spin_for(self, secs):
        end = time.monotonic() + secs
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def pose(self, frame):
        """base_footprint in `frame`: (x, y, yaw), or None."""
        try:
            t = self.tf.lookup_transform(frame, self.a.base_frame, Time())
        except TransformException as e:
            self.get_logger().warn(
                f"no TF {frame} -> {self.a.base_frame}: {e}", throttle_duration_sec=2.0
            )
            return None
        return t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)

    def wait_for_pose(self, frame, secs=10.0):
        end = time.monotonic() + secs
        while rclpy.ok() and time.monotonic() < end:
            self.spin_for(0.1)
            p = self.pose(frame)
            if p is not None:
                return p
        return None

    def stop(self):
        # Ctrl-C included: rclpy's own SIGINT handler is off (main), so the context is still
        # valid here and the zero command goes out rather than the last step command lingering.
        for _ in range(5):
            try:
                self.cmd.publish(Twist())
            except Exception:  # noqa: BLE001  (context already gone: nothing left to stop)
                return
            time.sleep(0.02)

    def errors(self, p, target):
        """Distance, and forward / left / turn errors in the robot's frame."""
        x, y, yaw = p
        tx, ty, tyaw = target
        dx, dy = tx - x, ty - y
        c, s = math.cos(yaw), math.sin(yaw)
        return (
            math.hypot(dx, dy),
            c * dx + s * dy,
            -s * dx + c * dy,
            wrap(tyaw - yaw),
            wrap(math.atan2(dy, dx) - yaw),
        )

    def report(self, e, prefix=""):
        dist, ex, ey, eyaw, _ = e
        self.get_logger().info(
            f"{prefix}{dist * 100:.1f} cm away: forward {ex * 100:+.1f} cm, left {ey * 100:+.1f} cm, "
            f"turn {math.degrees(eyaw):+.1f} deg"
        )

    def inside(self, e):
        return e[0] <= self.a.tol and abs(e[3]) <= math.radians(self.a.tol_yaw_deg)

    def walk(self, frame, target):
        a = self.a
        p = self.wait_for_pose(frame)
        if p is None:
            self.get_logger().error(f"no TF from {frame} to {a.base_frame}: is the odometry up?")
            return False
        e = self.errors(p, target)
        self.report(e, "start: ")
        if e[0] > 3.0:
            self.get_logger().warn(
                "more than 3 m to go, in a straight line with no obstacle avoidance"
            )
        tol, tyaw = a.tol, math.radians(a.tol_yaw_deg)
        inside_since, last_log = None, 0.0
        deadline = time.monotonic() + a.timeout
        while rclpy.ok():
            if time.monotonic() > deadline:
                self.stop()
                self.get_logger().error("gave up: not inside tolerance in time")
                return False
            self.spin_for(0.05)
            p = self.pose(frame)
            if p is None:
                self.stop()
                continue
            e = self.errors(p, target)
            dist, ex, ey, eyaw, bearing = e
            t = Twist()
            if dist > a.near:
                # Far: face the stance and walk to it, steering; sideways only near it.
                if abs(bearing) > math.radians(25.0):
                    t.angular.z = command(bearing, a.gain_yaw, 0.12, a.max_turn, 0.0)
                else:
                    t.linear.x = command(dist, a.gain, 0.06, a.max_speed_far, 0.0)
                    t.angular.z = command(bearing, a.gain_yaw, 0.12, a.max_turn, math.radians(3.0))
            else:
                t.linear.x = command(ex, a.gain, 0.06, a.max_speed, tol / math.sqrt(2.0))
                t.linear.y = command(ey, a.gain, 0.06, a.max_speed, tol / math.sqrt(2.0))
                t.angular.z = command(eyaw, a.gain_yaw, 0.12, a.max_turn, tyaw)
            if self.inside(e) or not (t.linear.x or t.linear.y or t.angular.z):
                self.cmd.publish(Twist())
                inside_since = inside_since or time.monotonic()
                if time.monotonic() - inside_since > a.settle:
                    self.stop()
                    self.report(e, "done: ")
                    return self.inside(e) or self._close_enough(e)
            else:
                inside_since = None
                self.cmd.publish(t)
            if time.monotonic() - last_log > 1.0:
                self.report(e)
                last_log = time.monotonic()
        return False

    def _close_enough(self, e):
        # Each axis inside its own share of the tolerance, the distance just over it: still fine.
        return e[0] <= 1.5 * self.a.tol and abs(e[3]) <= math.radians(self.a.tol_yaw_deg)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("stance", nargs="?", default=None, help="stance name (see --list)")
    p.add_argument("--x", type=float, help="target x, m (instead of a named stance)")
    p.add_argument("--y", type=float, help="target y, m")
    p.add_argument("--yaw-deg", type=float, default=0.0, help="target heading, degrees")
    p.add_argument("--frame", default=None, help="frame of --x/--y (default odom)")
    p.add_argument("--list", action="store_true", help="print the known stances")
    p.add_argument("--check", action="store_true", help="only report how far off the stance it is")
    p.add_argument(
        "--save", metavar="NAME", help=f"save where it stands now as NAME in {USER_FILE}"
    )
    p.add_argument("--tol", type=float, default=0.02, help="position tolerance, m")
    p.add_argument("--tol-yaw-deg", type=float, default=3.0, help="heading tolerance, degrees")
    p.add_argument("--near", type=float, default=1.0, help="m: nearer, move sideways too")
    p.add_argument("--gain", type=float, default=0.8, help="m/s per m of error")
    p.add_argument("--gain-yaw", type=float, default=1.0, help="rad/s per rad of error")
    p.add_argument("--max-speed", type=float, default=0.15, help="m/s near the stance")
    p.add_argument("--max-speed-far", type=float, default=0.30, help="m/s further away")
    p.add_argument("--max-turn", type=float, default=0.40, help="rad/s")
    p.add_argument("--settle", type=float, default=2.0, help="s inside tolerance to finish")
    p.add_argument("--timeout", type=float, default=180.0, help="s for the walk")
    p.add_argument("--base-frame", default="base_footprint")
    a = p.parse_args()

    stances = load_stances()
    if a.list:
        for name, s in sorted(stances.items()):
            print(
                f"{name}: x {s['x']:.3f}, y {s['y']:.3f}, heading {s.get('yaw_deg', 0.0):.1f} deg "
                f"in {s.get('frame', 'odom')}"
            )
        return

    # No rclpy SIGINT handler: Ctrl-C raises KeyboardInterrupt with the context still up, so
    # stop() can still command zero velocity.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    n = GoToStance(a)
    ok = True
    try:
        if a.save:
            frame = a.frame or "odom"
            pose = n.wait_for_pose(frame)
            if pose is None:
                n.get_logger().error(f"no TF from {frame} to {a.base_frame}")
                ok = False
            else:
                data = {}
                if os.path.exists(USER_FILE):
                    with open(USER_FILE) as f:
                        data = yaml.safe_load(f) or {}
                data.setdefault("stances", {})[a.save] = {
                    "x": round(pose[0], 3),
                    "y": round(pose[1], 3),
                    "yaw_deg": round(math.degrees(pose[2]), 1),
                    "frame": frame,
                }
                os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
                with open(USER_FILE, "w") as f:
                    yaml.safe_dump(data, f, sort_keys=True)
                n.get_logger().info(f"saved '{a.save}': {data['stances'][a.save]} in {USER_FILE}")
        else:
            if a.x is not None and a.y is not None:
                frame, target, name = (
                    a.frame or "odom",
                    (a.x, a.y, math.radians(a.yaw_deg)),
                    "the given pose",
                )
            elif a.stance and a.stance in stances:
                s = stances[a.stance]
                frame = a.frame or s.get("frame", "odom")
                target, name = (
                    (s["x"], s["y"], math.radians(s.get("yaw_deg", 0.0))),
                    f"'{a.stance}'",
                )
            else:
                n.get_logger().error(
                    f"unknown stance {a.stance!r}; known: {', '.join(sorted(stances)) or 'none'} "
                    "(or give --x --y [--yaw-deg])"
                )
                sys.exit(2)
            n.get_logger().info(
                f"stance {name}: ({target[0]:.3f}, {target[1]:.3f}) facing {math.degrees(target[2]):.1f} deg in {frame}"
            )
            if a.check:
                pose = n.wait_for_pose(frame)
                if pose is None:
                    ok = False
                else:
                    e = n.errors(pose, target)
                    n.report(e)
                    ok = n.inside(e)
                    n.get_logger().info("inside tolerance" if ok else "OUTSIDE tolerance")
            else:
                ok = n.walk(frame, target)
    except KeyboardInterrupt:
        ok = False
    finally:
        n.stop()
        n.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
