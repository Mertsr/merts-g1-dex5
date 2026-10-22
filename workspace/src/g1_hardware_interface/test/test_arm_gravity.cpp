/**
 * @file test_arm_gravity.cpp
 * @brief ArmGravity against a two-link pendulum whose holding torques are known by hand.
 */

#include <gmock/gmock.h>

#include <cmath>
#include <string>

#include "g1_hardware_interface/arm_gravity.hpp"

namespace
{
using g1_hardware_interface::ArmGravity;

// root --shoulder (about y)--> upper (2 kg, COM 0.5 m along x) --wrist (about y, at x 1 m)-->
// hand (1 kg, COM 0.2 m along x).
const char* const kUrdf = R"(<?xml version="1.0"?>
<robot name="pendulum">
  <link name="pelvis"/>
  <link name="upper">
    <inertial><origin xyz="0.5 0 0"/><mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>
  <link name="hand">
    <inertial><origin xyz="0.2 0 0"/><mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="pelvis"/><child link="upper"/><origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="10" velocity="1"/>
  </joint>
  <joint name="wrist" type="revolute">
    <parent link="upper"/><child link="hand"/><origin xyz="1 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="10" velocity="1"/>
  </joint>
</robot>)";

constexpr double kG = 9.81;

ArmGravity loaded(double hand_scale = 1.0)
{
    ArmGravity g;
    EXPECT_EQ(
        g.load(kUrdf, "pelvis", { { "shoulder", 0 }, { "wrist", 1 } }, { "hand" }, hand_scale),
        "");
    return g;
}

std::array<double, ArmGravity::kNumMotors>
torques(ArmGravity& g, double q0, double q1, std::array<double, 4> quat)
{
    std::array<double, ArmGravity::kNumMotors> q{};
    q[0] = q0;
    q[1] = q1;
    std::array<double, ArmGravity::kNumMotors> tau{};
    g.compute(q, quat, tau);
    return tau;
}
}  // namespace

TEST(ArmGravity, HorizontalArmNeedsTheMomentOfEverythingBeyondEachJoint)
{
    auto       g   = loaded();
    const auto tau = torques(g, 0.0, 0.0, { 1, 0, 0, 0 });
    // A positive turn about +y takes +x down, so gravity pushes that way and the motor holds
    // with the opposite: shoulder 2*0.5 + 1*1.2 kg m, wrist 1*0.2 kg m.
    EXPECT_NEAR(tau[0], -kG * (2.0 * 0.5 + 1.0 * 1.2), 1e-9);
    EXPECT_NEAR(tau[1], -kG * 0.2, 1e-9);
}

TEST(ArmGravity, HangingArmNeedsNothing)
{
    auto       g   = loaded();
    const auto tau = torques(g, M_PI / 2, 0.0, { 1, 0, 0, 0 });
    EXPECT_NEAR(tau[0], 0.0, 1e-9);
    EXPECT_NEAR(tau[1], 0.0, 1e-9);
}

TEST(ArmGravity, RootOrientationTurnsGravity)
{
    // Root pitched so its +x points straight up: gravity runs along the arm, which then needs no
    // torque at either joint.
    auto         g   = loaded();
    const double h   = std::sqrt(0.5);
    const auto   tau = torques(g, 0.0, 0.0, { h, 0, -h, 0 });
    EXPECT_NEAR(tau[0], 0.0, 1e-9);
    EXPECT_NEAR(tau[1], 0.0, 1e-9);
}

TEST(ArmGravity, ScaledSubtreeWeighsLess)
{
    auto       g   = loaded(0.25);
    const auto tau = torques(g, 0.0, 0.0, { 1, 0, 0, 0 });
    EXPECT_NEAR(tau[0], -kG * (2.0 * 0.5 + 0.25 * 1.2), 1e-9);
    EXPECT_NEAR(tau[1], -kG * 0.25 * 0.2, 1e-9);
    EXPECT_NEAR(g.totalMass(), 2.25, 1e-9);
}

TEST(ArmGravity, UnknownJointIsRefused)
{
    ArmGravity g;
    EXPECT_NE(g.load(kUrdf, "pelvis", { { "elbow", 3 } }), "");
    EXPECT_FALSE(g.loaded());
}
