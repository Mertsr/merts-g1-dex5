/**
 * @file test_dex5_sim_model.cpp
 * @brief The Dex5 simulated backend's kinematics: init, smooth approach, limits, finite state.
 */

#include <gmock/gmock.h>

#include <cmath>
#include <limits>

#include "g1_hand_interface/dex5_sim_model.hpp"

using g1_hand_interface::dex5JointName;
using g1_hand_interface::Dex5SimModel;
using g1_hand_interface::kDex5JointSuffixes;
using g1_hand_interface::kNumDex5Joints;

namespace
{
// The left hand's URDF limits, in the pinned order.
const Dex5SimModel::Array kLower = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
const Dex5SimModel::Array kUpper = { 1.658, 0.62, 1.4381, 1.4381, 1.4381, 1.4381 };

Dex5SimModel makeModel()
{
    Dex5SimModel model;
    EXPECT_TRUE(model.configure(kLower, kUpper, 3.0, 0.1));
    return model;
}
}  // namespace

TEST(Dex5SimModel, JointOrderIsThumbThenIndexMiddleRingLittleActiveJointsOnly)
{
    ASSERT_EQ(kNumDex5Joints, 6U);
    EXPECT_EQ(dex5JointName("left", 0), "left_thumb_1_joint");
    EXPECT_EQ(dex5JointName("left", 1), "left_thumb_2_joint");
    EXPECT_EQ(dex5JointName("left", 2), "left_index_1_joint");
    EXPECT_EQ(dex5JointName("left", 3), "left_middle_1_joint");
    EXPECT_EQ(dex5JointName("left", 4), "left_ring_1_joint");
    EXPECT_EQ(dex5JointName("left", 5), "left_little_1_joint");
    // The distal mimics (thumb_3, thumb_4, *_2) must never become claimable names.
    for (const char* suffix : kDex5JointSuffixes)
    {
        EXPECT_THAT(
            std::string(suffix),
            ::testing::Not(::testing::AnyOf(
                ::testing::HasSubstr("thumb_3"),
                ::testing::HasSubstr("thumb_4"),
                ::testing::HasSubstr("index_2"),
                ::testing::HasSubstr("middle_2"),
                ::testing::HasSubstr("ring_2"),
                ::testing::HasSubstr("little_2"))));
    }
}

TEST(Dex5SimModel, ResetClampsIntoLimitsAndStartsAtRest)
{
    Dex5SimModel model = makeModel();
    model.reset({ -1.0, 5.0, std::nan(""), 0.3, 0.3, 0.3 });
    EXPECT_EQ(model.position()[0], 0.0);
    EXPECT_EQ(model.position()[1], 0.62);
    EXPECT_EQ(model.position()[2], 0.0);
    EXPECT_EQ(model.position()[3], 0.3);
    for (const double v : model.velocity())
    {
        EXPECT_EQ(v, 0.0);
    }
}

TEST(Dex5SimModel, CommandEqualToStateMovesNothing)
{
    Dex5SimModel model = makeModel();
    model.reset({ 0.2, 0.1, 0.3, 0.4, 0.5, 0.6 });
    const auto before = model.position();
    model.step(before, 0.005);
    EXPECT_EQ(model.position(), before);
    for (const double v : model.velocity())
    {
        EXPECT_EQ(v, 0.0);
    }
}

TEST(Dex5SimModel, ApproachesTheCommandSmoothlyWithoutOvershoot)
{
    Dex5SimModel              model          = makeModel();
    const Dex5SimModel::Array command        = { 1.0, 0.5, 1.2, 1.2, 1.2, 1.2 };
    double                    previous_error = std::abs(command[0] - model.position()[0]);
    double                    previous_step  = 0.0;
    for (int tick = 0; tick < 400; ++tick)
    {
        const double before = model.position()[0];
        model.step(command, 0.005);
        const double step  = model.position()[0] - before;
        const double error = std::abs(command[0] - model.position()[0]);
        EXPECT_LE(error, previous_error);      // never moves away or overshoots
        EXPECT_GE(step, 0.0);                  // monotonic toward the command
        EXPECT_LE(step, 3.0 * 0.005 + 1e-12);  // honors the speed cap
        EXPECT_FALSE(tick > 0 && step > previous_step + 1e-12 && error < 0.5);  // decelerating
        previous_error = error;
        previous_step  = step;
    }
    for (std::size_t i = 0; i < kNumDex5Joints; ++i)
    {
        EXPECT_NEAR(model.position()[i], command[i], 1e-3) << i;
    }
}

TEST(Dex5SimModel, CommandsBeyondTheLimitsAreClamped)
{
    Dex5SimModel model = makeModel();
    for (int tick = 0; tick < 2000; ++tick)
    {
        model.step({ 9.0, 9.0, 9.0, 9.0, 9.0, 9.0 }, 0.005);
    }
    for (std::size_t i = 0; i < kNumDex5Joints; ++i)
    {
        EXPECT_LE(model.position()[i], kUpper[i]) << i;
        EXPECT_NEAR(model.position()[i], kUpper[i], 1e-3) << i;
    }
    for (int tick = 0; tick < 2000; ++tick)
    {
        model.step({ -9.0, -9.0, -9.0, -9.0, -9.0, -9.0 }, 0.005);
    }
    for (std::size_t i = 0; i < kNumDex5Joints; ++i)
    {
        EXPECT_GE(model.position()[i], kLower[i]) << i;
    }
}

TEST(Dex5SimModel, VelocityIsTheDeterministicPositionDifference)
{
    Dex5SimModel a = makeModel();
    Dex5SimModel b = makeModel();
    for (int tick = 0; tick < 50; ++tick)
    {
        const double before = a.position()[2];
        a.step({ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 }, 0.01);
        b.step({ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 }, 0.01);
        EXPECT_NEAR(a.velocity()[2], (a.position()[2] - before) / 0.01, 1e-12);
        EXPECT_LE(std::abs(a.velocity()[2]), 3.0 + 1e-12);
    }
    EXPECT_EQ(a.position(), b.position());
    EXPECT_EQ(a.velocity(), b.velocity());
}

TEST(Dex5SimModel, NonFiniteCommandsAndBadPeriodsKeepStateFinite)
{
    Dex5SimModel model = makeModel();
    model.reset({ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 });
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    model.step({ nan, inf, -inf, 0.5, 0.5, 0.5 }, 0.005);
    EXPECT_EQ(model.position()[0], 0.1);  // held
    for (const double dt : { 0.0, -1.0, nan, inf })
    {
        const auto before = model.position();
        model.step({ 1.0, 0.5, 1.0, 1.0, 1.0, 1.0 }, dt);
        EXPECT_EQ(model.position(), before) << dt;
    }
    model.step({ 1.0, 0.5, 1.0, 1.0, 1.0, 1.0 }, 1000.0);  // a stalled loop cannot jump
    for (std::size_t i = 0; i < kNumDex5Joints; ++i)
    {
        EXPECT_TRUE(std::isfinite(model.position()[i]));
        EXPECT_TRUE(std::isfinite(model.velocity()[i]));
        EXPECT_LE(model.position()[i], 0.1 + 0.015 + 3.0 * 0.1 + 1e-9);
    }
}

TEST(Dex5SimModel, StopZeroesVelocityAndKeepsPosition)
{
    Dex5SimModel model = makeModel();
    model.step({ 1.0, 0.5, 1.0, 1.0, 1.0, 1.0 }, 0.01);
    const auto position = model.position();
    model.stop();
    EXPECT_EQ(model.position(), position);
    for (const double v : model.velocity())
    {
        EXPECT_EQ(v, 0.0);
    }
}

TEST(Dex5SimModel, ConfigureRefusesUnsafeValuesAndStaysUnchanged)
{
    Dex5SimModel model      = makeModel();
    auto         transposed = kLower;
    transposed[3]           = 2.0;
    EXPECT_FALSE(model.configure(transposed, kUpper, 3.0, 0.1));
    EXPECT_FALSE(model.configure(kLower, kLower, 3.0, 0.1));
    EXPECT_FALSE(model.configure(kLower, kUpper, 0.0, 0.1));
    EXPECT_FALSE(model.configure(kLower, kUpper, 3.0, 0.0));
    EXPECT_FALSE(model.configure(kLower, kUpper, std::nan(""), 0.1));
    EXPECT_EQ(model.upper(), kUpper);
}
