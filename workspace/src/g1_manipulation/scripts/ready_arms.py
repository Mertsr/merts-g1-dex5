#!/usr/bin/env python3
"""Put both arms and hands in the ready posture a pick starts from.

    ros2 run g1_manipulation ready_arms.py              # arms tucked, hands open
    ros2 run g1_manipulation ready_arms.py --hands none # arms only
    ros2 run g1_manipulation ready_arms.py --arms none --hands open

Run it before walking up to the table (go_to_stance.py) and before every pick: the pick plans
from wherever the arms are, and an arm left out in front -- after a failed pick, say -- is what
blocks the other hand's lift or swings into the table.

THE ARMS, tucked: hanging beside the hips, elbows a little bent (shoulder pitch 0.3, roll 0.2
outward, elbow 1.2), the hands 19 cm out to each side and 29 cm below the pelvis. Out of the
table's way (its edge is about 30 cm ahead of the pelvis at the stance), out of the lifting
hand's way, and the start the scoop's reach was checked from.

THE HANDS, open: every finger and the thumb straight (g1.srdf's "open"); the pick opens each hand
to its grasp's own shape (scoop_open) itself.

Both through the controllers MoveIt executes on (/arm_trajectory_controller and the two hand
controllers, FollowJointTrajectory), so it needs the arm active (activate_arm:=true and its
delay passed). A pick that failed half-way may have left the arm in contact with the table:
the arms move slowly (--time, 5 s) and straight in joint space, so check the way is clear.
"""

import argparse
import sys

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

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
ARMS = {
    "tucked": [0.3, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0, 0.3, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0],
}
HAND_JOINTS = ["thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "little_1"]
HANDS = {
    "open": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "scoop_open": [1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
}


class ReadyArms(Node):
    def __init__(self):
        super().__init__("ready_arms")

    def move(self, controller, joints, positions, secs):
        client = ActionClient(self, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory")
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f"no /{controller}/follow_joint_trajectory: is the arm active "
                "(activate_arm:=true, and its delay passed)?"
            )
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joints
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.velocities = [0.0] * len(positions)
        pt.time_from_start = Duration(sec=int(secs), nanosec=int((secs % 1) * 1e9))
        goal.trajectory.points = [pt]
        fut = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        handle = fut.result()
        if not handle or not handle.accepted:
            self.get_logger().error(f"{controller}: the trajectory was rejected")
            return False
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res)
        code = res.result().result.error_code
        if code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(f"{controller}: ended with error code {code}")
            return False
        return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arms", default="tucked", choices=sorted(ARMS) + ["none"])
    p.add_argument("--hands", default="open", choices=sorted(HANDS) + ["none"])
    p.add_argument("--time", type=float, default=5.0, help="s for the arm move")
    a = p.parse_args()

    rclpy.init()
    n = ReadyArms()
    ok = True
    try:
        # Hands first: open fingers do not catch on anything while the arms come in.
        if a.hands != "none":
            for side in ("left", "right"):
                ok = (
                    n.move(
                        f"{side}_hand_controller",
                        [f"{side}_{j}_joint" for j in HAND_JOINTS],
                        HANDS[a.hands],
                        1.5,
                    )
                    and ok
                )
            n.get_logger().info(f"hands: {a.hands}" if ok else "hands: FAILED")
        if a.arms != "none" and ok:
            ok = n.move("arm_trajectory_controller", ARM_JOINTS, ARMS[a.arms], a.time)
            n.get_logger().info(f"arms: {a.arms}" if ok else "arms: FAILED")
    except KeyboardInterrupt:
        ok = False
    finally:
        n.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
