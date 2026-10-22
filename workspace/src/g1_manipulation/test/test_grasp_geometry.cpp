/**
 * @file test_grasp_geometry.cpp
 * @brief The arm/group mapping and the grasp-frame goal, without a running MoveIt.
 *
 * Where the hand grips is no longer arithmetic here at all: it is the {side}_hand_grasp_frame
 * link in g1_description, and goals are given for that frame directly. What is left to pin is
 * which groups and frames an "arm" string resolves to, and that the goal passes position
 * through untouched while orienting the closing axis at the floor.
 */

#include <gmock/gmock.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>

#include <cmath>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <vector>

#include "g1_manipulation/g1_manipulation_server_node.hpp"

using g1_manipulation::ArmContext;
using g1_manipulation::resolveArm;

namespace
{

/// The server's own graspFrameGoal, which is a private member. Position passes through; only
/// the orientation is computed, and it mirrors between hands.
geometry_msgs::msg::Pose
graspFrameGoal(const geometry_msgs::msg::Pose& object, const std::vector<double>& rpy, bool is_left)
{
    geometry_msgs::msg::Pose goal;
    goal.position = object.position;
    tf2::Quaternion rotation;
    rotation.setRPY((is_left ? -1.0 : 1.0) * rpy[0], rpy[1], rpy[2]);
    goal.orientation = tf2::toMsg(rotation);
    return goal;
}

geometry_msgs::msg::Pose objectAt(double x, double y, double z)
{
    geometry_msgs::msg::Pose pose;
    pose.position.x    = x;
    pose.position.y    = y;
    pose.position.z    = z;
    pose.orientation.w = 1.0;
    return pose;
}

}  // namespace

TEST(ResolveArm, MapsASideOntoItsGroupsFramesAndHandedness)
{
    ArmContext arm;
    ASSERT_TRUE(resolveArm("left", arm));
    EXPECT_EQ(arm.arm_group, "left_arm");
    EXPECT_EQ(arm.hand_group, "left_hand");
    EXPECT_EQ(arm.palm_link, "left_hand_palm_link");
    EXPECT_EQ(arm.grasp_frame, "left_hand_grasp_frame");
    EXPECT_TRUE(arm.is_left);

    ASSERT_TRUE(resolveArm("right", arm));
    EXPECT_EQ(arm.arm_group, "right_arm");
    EXPECT_EQ(arm.grasp_frame, "right_hand_grasp_frame");
    EXPECT_FALSE(arm.is_left);
}

TEST(ResolveArm, RejectsAnythingElseWithoutAssigning)
{
    // The names have to match g1.srdf's groups and the URDF's frames exactly. A silent
    // fallback to one side would move the wrong arm, which is what this refusal prevents.
    ArmContext arm;
    arm.arm_group = "sentinel";
    EXPECT_FALSE(resolveArm("Left", arm));
    EXPECT_FALSE(resolveArm("left_arm", arm));
    EXPECT_FALSE(resolveArm("", arm));
    EXPECT_EQ(arm.arm_group, "sentinel");
}

TEST(GraspFrameGoal, PutsTheGraspFrameExactlyOnTheObject)
{
    // The whole point of moving the offset into the URDF: the goal position IS the object
    // position, with no arithmetic left in this file to get wrong.
    const std::vector<double> rpy{ -M_PI_2, 0.0, 0.0 };
    const auto                target = objectAt(0.35, -0.20, 0.83);

    const auto goal = graspFrameGoal(target, rpy, /*is_left=*/false);

    EXPECT_DOUBLE_EQ(goal.position.x, target.position.x);
    EXPECT_DOUBLE_EQ(goal.position.y, target.position.y);
    EXPECT_DOUBLE_EQ(goal.position.z, target.position.z);
}

TEST(GraspFrameGoal, PointsTheClosingAxisAtTheFloor)
{
    // The hand's fingers curl toward the palm's +y, so it is THAT axis that has to end up
    // pointing down for a grasp off a table -- and the palm's +x has to stay forward, so the
    // arm reaches out rather than the wrist contorting. Getting this wrong is easy to miss by
    // inspection, so it is pinned here instead.
    const auto palm = graspFrameGoal(objectAt(0.4, 0.0, 0.8), { -M_PI_2, 0.0, 0.0 }, false);

    tf2::Quaternion rotation;
    tf2::fromMsg(palm.orientation, rotation);
    const tf2::Matrix3x3 basis(rotation);

    EXPECT_NEAR((basis * tf2::Vector3(0.0, 1.0, 0.0)).z(), -1.0, 1e-9)
        << "the palm's +y, where the fingers close, must point down";
    EXPECT_NEAR((basis * tf2::Vector3(1.0, 0.0, 0.0)).x(), 1.0, 1e-9)
        << "and the palm's +x stays forward";
}

TEST(GraspFrameGoal, TheTwoHandsMirror)
{
    // The hands are mirror images, so the roll that points one hand's closing axis down points
    // the other's up. A shared orientation would have the left hand grasping upside down.
    const auto right = graspFrameGoal(objectAt(0.35, 0.20, 0.83), { -M_PI_2, 0.0, 0.0 }, false);
    const auto left  = graspFrameGoal(objectAt(0.35, 0.20, 0.83), { -M_PI_2, 0.0, 0.0 }, true);

    tf2::Quaternion qr;
    tf2::Quaternion ql;
    tf2::fromMsg(right.orientation, qr);
    tf2::fromMsg(left.orientation, ql);

    EXPECT_NEAR((tf2::Matrix3x3(qr) * tf2::Vector3(0.0, 1.0, 0.0)).z(), -1.0, 1e-9);
    EXPECT_NEAR((tf2::Matrix3x3(ql) * tf2::Vector3(0.0, -1.0, 0.0)).z(), -1.0, 1e-9)
        << "the left hand closes toward its own -y, so its -y is what must point down";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}

// A published object grasp is the RIGHT hand's grasp frame; the left takes it turned half a turn
// about the fingers, so both hands' fingers point the same way and close toward the same side.
TEST(ObjectGrasp, LeftHandMirrorsAboutTheFingers)
{
    geometry_msgs::msg::Pose grasp = objectAt(0.3, 0.2, 0.1);
    tf2::Quaternion          q;
    q.setRPY(0.3, -0.2, 0.7);
    grasp.orientation = tf2::toMsg(q);

    const auto right = g1_manipulation::handGoalFromGrasp(grasp, false);
    const auto left  = g1_manipulation::handGoalFromGrasp(grasp, true);
    EXPECT_DOUBLE_EQ(right.position.x, 0.3);
    EXPECT_DOUBLE_EQ(left.position.z, 0.1);

    tf2::Quaternion qr, ql;
    tf2::fromMsg(right.orientation, qr);
    tf2::fromMsg(left.orientation, ql);
    const tf2::Matrix3x3 R(qr), L(ql);
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(L.getColumn(0)[i], R.getColumn(0)[i], 1e-9);   // fingers: same direction
        EXPECT_NEAR(L.getColumn(1)[i], -R.getColumn(1)[i], 1e-9);  // left closes toward its -y
    }
}

TEST(ObjectGrasp, BackOffRetreatsAgainstTheFingers)
{
    geometry_msgs::msg::Pose grasp = objectAt(0.3, 0.0, 0.1);
    tf2::Quaternion          down;
    down.setRPY(0.0, M_PI_2, 0.0);  // fingers (x) pointing at the floor
    grasp.orientation = tf2::toMsg(down);
    const auto pre    = g1_manipulation::backOff(grasp, 0.12);
    EXPECT_NEAR(pre.position.x, 0.3, 1e-9);
    EXPECT_NEAR(pre.position.z, 0.22, 1e-9);  // straight up: the approach comes down
}

// Variants of a published grasp keep the point and the closing axis, and only turn the fingers in
// the plane across it; the published orientation comes first, and nothing points the fingers up.
TEST(ObjectGrasp, VariantsTurnAboutTheClosingAxis)
{
    geometry_msgs::msg::Pose grasp = objectAt(0.3, -0.1, 0.02);
    tf2::Quaternion          down;
    down.setRPY(0.0, M_PI_2, 0.0);  // fingers at the floor, closing toward +y
    grasp.orientation = tf2::toMsg(down);

    const auto variants = g1_manipulation::graspVariants(grasp, M_PI_2, M_PI / 12, true, 0.25);
    ASSERT_FALSE(variants.empty());
    EXPECT_DOUBLE_EQ(variants.front().tilt, 0.0);
    EXPECT_FALSE(variants.front().flipped);
    EXPECT_EQ(variants.front().suffix(), "");

    tf2::Quaternion q0;
    tf2::fromMsg(grasp.orientation, q0);
    const tf2::Vector3 closing = tf2::Matrix3x3(q0).getColumn(1);
    int                flipped = 0;
    for (const auto& v : variants)
    {
        EXPECT_DOUBLE_EQ(v.pose.position.x, 0.3);
        EXPECT_DOUBLE_EQ(v.pose.position.z, 0.02);
        tf2::Quaternion q;
        tf2::fromMsg(v.pose.orientation, q);
        const tf2::Matrix3x3 R(q);
        EXPECT_NEAR(std::abs(R.getColumn(1).dot(closing)), 1.0, 1e-9);  // same closing axis
        EXPECT_LE(R.getColumn(0).z(), -0.25 + 1e-9);                      // fingers coming down
        flipped += v.flipped ? 1 : 0;
    }
    // cos(tilt) >= 0.25 keeps |tilt| <= 75 deg: 0, +-15 .. +-75, each way up.
    EXPECT_EQ(variants.size(), 22u);
    EXPECT_EQ(flipped, 11);
}

namespace
{
/// The scoop as published: fingers along +x pitched `pitch` down, palm (y) up.
geometry_msgs::msg::Pose scoopAt(double pitch, double roll = 0.0)
{
    geometry_msgs::msg::Pose pose = objectAt(0.4, -0.2, 0.0);
    tf2::Quaternion          q;
    // Rx(-90 deg) takes y to z: palm up, fingers along +x; then pitched about the world's y.
    tf2::Quaternion pitch_q, roll_q, base;
    base.setRPY(M_PI_2, 0.0, 0.0);
    pitch_q.setRPY(0.0, pitch, 0.0);
    roll_q.setRPY(roll, 0.0, 0.0);
    q                = pitch_q * roll_q * base;
    pose.orientation = tf2::toMsg(q.normalized());
    return pose;
}
tf2::Matrix3x3 rotOf(const geometry_msgs::msg::Pose& pose)
{
    tf2::Quaternion q;
    tf2::fromMsg(pose.orientation, q);
    return tf2::Matrix3x3(q);
}
}  // namespace

// A panel estimated tilted tips the published scoop with it; levelScoop puts it back to the
// configured pitch below the world's horizontal, palm up, the fingers' heading and point kept.
TEST(Scoop, LevelScoopRestoresPitchAndRoll)
{
    const double       pitch  = 25.0 * M_PI / 180.0;
    const auto         tilted = scoopAt(5.0 * M_PI / 180.0, 0.3);  // flatter and rolled
    const tf2::Vector3 up(0.0, 0.0, 1.0);
    const auto         level = g1_manipulation::levelScoop(tilted, up, pitch);
    EXPECT_NEAR(g1_manipulation::fingerPitch(level.orientation, up), pitch, 1e-9);
    EXPECT_DOUBLE_EQ(level.position.x, tilted.position.x);
    const tf2::Matrix3x3 R = rotOf(level);
    EXPECT_NEAR(R.getColumn(0).y(), 0.0, 1e-9);  // heading kept: along +x
    EXPECT_GT(R.getColumn(0).x(), 0.0);
    EXPECT_NEAR(R.getColumn(2).z(), 0.0, 1e-9);  // no roll: the closing axis is level
    EXPECT_GT(R.getColumn(1).z(), 0.8);          // palm up
}

// A left hand's goal is the scoop turned over about the fingers (handGoalFromGrasp); levelling
// keeps it turned over.
TEST(Scoop, LevelScoopKeepsALeftHandTurnedOver)
{
    const tf2::Vector3 up(0.0, 0.0, 1.0);
    const auto         left  = g1_manipulation::handGoalFromGrasp(scoopAt(0.4), true);
    const auto         level = g1_manipulation::levelScoop(left, up, 0.4363);
    EXPECT_LT(rotOf(level).getColumn(1).z(), -0.8);
    EXPECT_NEAR(g1_manipulation::fingerPitch(level.orientation, up), 0.4363, 1e-9);
}

// With the pelvis leaning, the slide runs level with the world, not the pelvis.
TEST(Scoop, SlideBackRunsLevelWithTheWorld)
{
    const tf2::Vector3 up    = tf2::Vector3(0.2, 0.0, 1.0).normalized();  // pelvis pitched
    const auto         grasp = g1_manipulation::levelScoop(scoopAt(0.4), up, 0.4363);
    const auto         back  = g1_manipulation::slideBack(grasp, 0.11, up);
    const tf2::Vector3 d(
        back.position.x - grasp.position.x,
        back.position.y - grasp.position.y,
        back.position.z - grasp.position.z);
    EXPECT_NEAR(d.length(), 0.11, 1e-9);
    EXPECT_NEAR(d.dot(up), 0.0, 1e-9);
    const auto above = g1_manipulation::raised(back, up, 0.08);
    EXPECT_NEAR(above.position.z - back.position.z, 0.08 * up.z(), 1e-9);
}
