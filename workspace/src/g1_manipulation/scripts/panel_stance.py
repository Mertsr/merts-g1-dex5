#!/usr/bin/env python3
"""Put the G1 where the two-hand scoop of a BSH panel works: arms tucked, base squared to the panel.

    python3 panel_stance.py                  # panel_1: arms tucked, then walk into the stance
    python3 panel_stance.py --object panel_2
    python3 panel_stance.py --arms-only      # just tuck the arms
    python3 panel_stance.py --no-arms        # just walk
    python3 panel_stance.py --check          # only print how far off the stance is now

Run it with the stack up (bringup.launch.py ... nav:=true manipulation:=true activate_arm:=true
object_tracker:=true), the robot standing a metre or two in front of the table, facing it and
the panel in the head camera's view. Then the pick:

    ros2 action send_goal /g1_manipulation_server/pick g1_msgs/action/Pick \\
        "{object_id: panel_1, arm: auto}" --feedback

THE STANCE. The scoop and the second hand's clamp were worked out in MuJoCo with the pelvis at
world (1.20, -0.80) facing +x, i.e. square to the panel, centred on its length, 20 cm back from
the table's edge (x 1.40), with part_tray_1 at rest at (1.595, -0.800). In the robot's ground frame
(base_footprint, the pelvis on the floor) the panel's origin (its tracked pose) is then at
x 0.395 m ahead, y 0.000 m, its long axis across the robot. The clamp held for stances 2-3 cm and
5 degrees off that, so the tolerances here are 1.5 cm and 3 degrees.

THE ARMS. Tucked: hanging beside the hips, elbows a little bent (shoulder pitch 0.3, roll 0.2
outward, elbow 1.2), the hands 19 cm out to each side and 29 cm below the pelvis, about level with
it front to back. Out of the table's way (its edge is 20 cm ahead of the pelvis, and a hand held
forward at waist height goes under its top), and out of the gripping hand's way when it lifts the
panel: an arm held in front of the body is what stopped the lift before. Done FIRST, before the
walk, so the arms are not swung near the table.

HOW IT WALKS. Closed loop on the tracker: the panel's pose on /objects, into base_footprint
through TF, filtered, 20 Hz. Forward, sideways and turn at once, each proportional to its error
and held above the gait shaper's dead zone (0.05 m/s, 0.10 rad/s) until inside tolerance, capped
at 0.15 m/s and 0.3 rad/s. Commands go to /cmd_vel_approach, the gait shaper's priority input
(what g1_base_approach uses), so it needs nav:=true. Done when it stands inside tolerance for 2 s.
"""

import argparse
import math
import sys
import time

import rclpy
import tf2_geometry_msgs  # noqa: F401  (registers PoseStamped with tf2)
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint
from vision_msgs.msg import Detection3DArray

ARM_JOINTS = [
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
]
# Same order as ARM_JOINTS. The right arm mirrors the left: roll and yaw change sign.
TUCKED = [0.3, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0, 0.3, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0]


def yaw_of(q):
    """Heading of a quaternion's x axis, projected on the floor."""
    x, y, z, w = q.x, q.y, q.z, q.w
    ax = 1.0 - 2.0 * (y * y + z * z)
    ay = 2.0 * (x * y + w * z)
    return math.atan2(ay, ax)


def axis_error(axis_angle, wanted):
    """axis_angle - wanted for an AXIS (no direction): wrapped into [-90, 90) degrees."""
    e = axis_angle - wanted
    return (e + math.pi / 2) % math.pi - math.pi / 2


def command(error, gain, floor, cap, tolerance):
    """Proportional, but at least `floor` (the gait's dead zone) while outside tolerance."""
    if abs(error) <= tolerance:
        return 0.0
    v = gain * error
    return math.copysign(min(max(abs(v), floor), cap), v)


class PanelStance(Node):
    def __init__(self, a):
        super().__init__("panel_stance")
        self.a = a
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.latest = None  # (x, y, axis angle) in base_footprint
        self.stamp = 0.0
        self.create_subscription(Detection3DArray, "/objects", self.on_objects, 10)
        self.cmd = self.create_publisher(Twist, "/cmd_vel_approach", 10)
        self.arm = ActionClient(
            self, FollowJointTrajectory, "/arm_trajectory_controller/follow_joint_trajectory"
        )

    # ---- perception -------------------------------------------------------------------------
    def on_objects(self, msg):
        for det in msg.detections:
            if not det.results or det.results[0].hypothesis.class_id != self.a.object:
                continue
            ps = PoseStamped()
            ps.header.frame_id = det.header.frame_id or msg.header.frame_id
            ps.header.stamp = Time().to_msg()  # latest TF: exact stamps fail while walking
            ps.pose = det.results[0].pose.pose
            try:
                p = self.tf.transform(ps, self.a.base_frame)
            except TransformException as e:
                self.get_logger().warn(
                    f"cannot transform the panel pose: {e}", throttle_duration_sec=2.0
                )
                return
            m = (p.pose.position.x, p.pose.position.y, yaw_of(p.pose.orientation))
            if self.latest is None or time.monotonic() - self.stamp > 1.0:
                self.latest = m
            else:
                k = 0.3  # light low-pass: the head sways with the gait
                lx, ly, la = self.latest
                da = axis_error(m[2], la)
                self.latest = (lx + k * (m[0] - lx), ly + k * (m[1] - ly), la + k * da)
            self.stamp = time.monotonic()
            return

    def errors(self):
        """Forward, sideways and turn errors (m, m, rad), or None if the panel is not seen."""
        if self.latest is None or time.monotonic() - self.stamp > 1.0:
            return None
        x, y, axis = self.latest
        # Square to the panel: its long axis (its body x) across the robot, i.e. at +-90 degrees.
        return x - self.a.x, y - self.a.y, axis_error(axis, math.pi / 2)

    def spin_for(self, secs):
        end = time.monotonic() + secs
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    # ---- arms -------------------------------------------------------------------------------
    def tuck_arms(self):
        if not self.arm.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "no /arm_trajectory_controller/follow_joint_trajectory: is the arm active "
                "(activate_arm:=true, and its delay passed)?"
            )
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = TUCKED
        pt.velocities = [0.0] * len(TUCKED)
        pt.time_from_start = Duration(
            sec=int(self.a.arm_time), nanosec=int((self.a.arm_time % 1) * 1e9)
        )
        goal.trajectory.points = [pt]
        self.get_logger().info(f"arms: tucking in {self.a.arm_time:.0f} s")
        fut = self.arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        handle = fut.result()
        if not handle or not handle.accepted:
            self.get_logger().error("arms: the controller rejected the trajectory")
            return False
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res)
        code = res.result().result.error_code
        if code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(f"arms: trajectory ended with error code {code}")
            return False
        self.get_logger().info("arms: tucked")
        return True

    # ---- base -------------------------------------------------------------------------------
    def stop(self):
        # Ctrl-C included: rclpy's own SIGINT handler is off (main), so the context is still
        # valid here and the zero command goes out rather than the last step command lingering.
        for _ in range(5):
            try:
                self.cmd.publish(Twist())
            except Exception:  # noqa: BLE001  (context already gone: nothing left to stop)
                return
            time.sleep(0.02)

    def report(self, e, prefix=""):
        ex, ey, ea = e
        self.get_logger().info(
            f"{prefix}panel {self.a.object}: forward {ex * 100:+.1f} cm, left {ey * 100:+.1f} cm, "
            f"turn {math.degrees(ea):+.1f} deg off the stance"
        )

    def wait_for_panel(self, secs=10.0):
        end = time.monotonic() + secs
        while rclpy.ok() and time.monotonic() < end:
            self.spin_for(0.1)
            e = self.errors()
            if e is not None:
                return e
        return None

    def walk(self):
        a = self.a
        e = self.wait_for_panel()
        if e is None:
            self.get_logger().error(
                f"'{a.object}' is not on /objects: is the tracker running and the panel in view?"
            )
            return False
        self.report(e, "start: ")
        tol, tyaw = a.tol, math.radians(a.tol_yaw_deg)
        inside_since = None
        deadline = time.monotonic() + a.timeout
        last_log = 0.0
        while rclpy.ok():
            if time.monotonic() > deadline:
                self.stop()
                self.get_logger().error("gave up: not inside tolerance in time")
                return False
            self.spin_for(0.05)
            e = self.errors()
            if e is None:
                self.stop()
                self.get_logger().warn(
                    "lost the panel; standing until it is seen again", throttle_duration_sec=2.0
                )
                continue
            ex, ey, ea = e
            t = Twist()
            t.linear.x = command(ex, a.gain, 0.06, a.max_speed, tol)
            t.linear.y = command(ey, a.gain, 0.06, a.max_speed, tol)
            t.angular.z = command(ea, a.gain_yaw, 0.12, a.max_turn, tyaw)
            moving = t.linear.x or t.linear.y or t.angular.z
            if moving:
                inside_since = None
                self.cmd.publish(t)
            else:
                self.cmd.publish(Twist())
                inside_since = inside_since or time.monotonic()
                if time.monotonic() - inside_since > a.settle:
                    self.stop()
                    self.report(e, "done: ")
                    return True
            if time.monotonic() - last_log > 1.0:
                self.report(e)
                last_log = time.monotonic()
        return False


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--object", default="panel_1", help="tracked panel to stand at (default panel_1)"
    )
    p.add_argument("--x", type=float, default=0.395, help="panel origin ahead of base_footprint, m")
    p.add_argument("--y", type=float, default=0.000, help="panel origin left of base_footprint, m")
    p.add_argument("--tol", type=float, default=0.015, help="position tolerance, m")
    p.add_argument("--tol-yaw-deg", type=float, default=3.0, help="heading tolerance, degrees")
    p.add_argument("--gain", type=float, default=0.8, help="m/s per m of error")
    p.add_argument("--gain-yaw", type=float, default=1.0, help="rad/s per rad of error")
    p.add_argument("--max-speed", type=float, default=0.15, help="m/s")
    p.add_argument("--max-turn", type=float, default=0.30, help="rad/s")
    p.add_argument("--settle", type=float, default=2.0, help="s inside tolerance to finish")
    p.add_argument("--timeout", type=float, default=120.0, help="s for the walk")
    p.add_argument("--arm-time", type=float, default=5.0, help="s for the arm tuck")
    p.add_argument("--base-frame", default="base_footprint")
    p.add_argument("--arms-only", action="store_true")
    p.add_argument("--no-arms", action="store_true")
    p.add_argument("--check", action="store_true", help="only report the current offset")
    a = p.parse_args()

    # No rclpy SIGINT handler: Ctrl-C raises KeyboardInterrupt with the context still up, so
    # stop() can still command zero velocity.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    n = PanelStance(a)
    ok = True
    try:
        if a.check:
            e = n.wait_for_panel()
            if e is None:
                n.get_logger().error(f"'{a.object}' is not on /objects")
                ok = False
            else:
                n.spin_for(1.0)  # let the filter settle
                e = n.errors() or e
                n.report(e)
                ok = (
                    abs(e[0]) <= a.tol
                    and abs(e[1]) <= a.tol
                    and abs(e[2]) <= math.radians(a.tol_yaw_deg)
                )
                n.get_logger().info("inside tolerance" if ok else "OUTSIDE tolerance")
        else:
            if not a.no_arms:
                ok = n.tuck_arms()
            if ok and not a.arms_only:
                ok = n.walk()
    except KeyboardInterrupt:
        ok = False
    finally:
        n.stop()
        n.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
