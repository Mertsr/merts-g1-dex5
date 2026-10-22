/**
 * @file test_walk_policy_isaaclab.cpp
 * @brief Pins the IsaacLab-sourced walking policy's wire contract: index tables, observation
 * layout, action mapping, history padding, and action delay -- all without a live node, DDS,
 * ONNX Runtime, or the sim.
 */
#include <gmock/gmock.h>

#include "g1_motion_service_sim/walk_policy_isaaclab.hpp"

namespace g1_motion_service_sim
{
namespace
{

using ::testing::FloatEq;

IlPolicyConfig makeConfig()
{
    IlPolicyConfig cfg;
    // Distinct per-joint values so an off-by-one in any index arithmetic shows up as a wrong
    // number rather than coincidentally matching its neighbour.
    for (std::size_t i = 0; i < kIlNumJoints; ++i)
    {
        cfg.default_joint_pos[i] = 0.01 * static_cast<double>(i);
        cfg.default_joint_vel[i] = 0.001 * static_cast<double>(i);
    }
    cfg.leg_action_indices = { 0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18 };
    cfg.arm_obs_indices    = { 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28 };
    cfg.action_scale       = 0.25;
    cfg.command_height     = 0.8;
    return cfg;
}

// kIlFromDds and kLegActionToDds must agree: kLegActionToDds[i] == kIlFromDds[leg_action_indices[i]].
// A hand-maintained shortcut table silently drifting from its source table is exactly the kind of
// bug checkJointOrder() exists to catch for walker.onnx; this test is the equivalent guard here.
TEST(WalkPolicyIsaacLab, LegActionToDdsMatchesFromDdsComposedWithLegIndices)
{
    const auto cfg = makeConfig();
    for (std::size_t i = 0; i < kIlNumLegs; ++i)
    {
        const auto il_index = static_cast<std::size_t>(cfg.leg_action_indices[i]);
        EXPECT_EQ(kLegActionToDds[i], kIlFromDds[il_index]) << "mismatch at action index " << i;
    }
}

// Every DDS index 0-28 must appear exactly once across kIlFromDds -- a permutation, not just 29
// values in range. A duplicate or a gap means two IsaacLab joints are reading the same DDS motor
// (or one DDS motor is never read at all) while every index individually still looks plausible.
TEST(WalkPolicyIsaacLab, FromDdsIsAPermutation)
{
    std::array<bool, kIlNumJoints> seen{};
    for (int dds : kIlFromDds)
    {
        ASSERT_GE(dds, 0);
        ASSERT_LT(static_cast<std::size_t>(dds), kIlNumJoints);
        ASSERT_FALSE(seen[static_cast<std::size_t>(dds)]) << "DDS index " << dds << " read twice";
        seen[static_cast<std::size_t>(dds)] = true;
    }
}

TEST(WalkPolicyIsaacLab, ObservationStepLayoutAndRelativeJointTerms)
{
    const auto cfg = makeConfig();

    IlPolicyInputs in{};
    in.base_ang_vel_body = { 1.0, 2.0, 3.0 };
    in.projected_gravity = { 4.0, 5.0, 6.0 };
    in.cmd_vx             = 0.5;
    in.cmd_vy             = -0.1;
    in.cmd_vyaw            = 0.2;
    for (std::size_t i = 0; i < kIlNumJoints; ++i)
    {
        // Offset from default by a known, distinct amount so the subtraction is checkable.
        in.joint_pos[i] = cfg.default_joint_pos[i] + 0.5;
        in.joint_vel[i] = cfg.default_joint_vel[i] + 0.25;
    }

    std::array<float, kIlNumJoints> last_action{};
    for (std::size_t i = 0; i < kIlNumJoints; ++i) last_action[i] = static_cast<float>(i) * 0.1F;

    const auto obs = ilAssembleObservationStep(in, cfg, last_action);

    EXPECT_THAT(obs[0], FloatEq(1.0F));
    EXPECT_THAT(obs[1], FloatEq(2.0F));
    EXPECT_THAT(obs[2], FloatEq(3.0F));
    EXPECT_THAT(obs[3], FloatEq(4.0F));
    EXPECT_THAT(obs[4], FloatEq(5.0F));
    EXPECT_THAT(obs[5], FloatEq(6.0F));
    EXPECT_THAT(obs[6], FloatEq(0.5F));
    EXPECT_THAT(obs[7], FloatEq(-0.1F));
    EXPECT_THAT(obs[8], FloatEq(0.2F));
    EXPECT_THAT(obs[9], FloatEq(0.8F));  // command_height

    // joint_pos block: legs (12) then arms (14), each relative to default -> every entry 0.5.
    for (std::size_t i = 10; i < 10 + kIlNumLegs + kIlNumArmsObs; ++i)
    {
        EXPECT_THAT(obs[i], FloatEq(0.5F)) << "joint_pos offset " << i;
    }
    // joint_vel block, same shape, relative offset 0.25.
    const std::size_t vel_start = 10 + kIlNumLegs + kIlNumArmsObs;
    for (std::size_t i = vel_start; i < vel_start + kIlNumLegs + kIlNumArmsObs; ++i)
    {
        EXPECT_THAT(obs[i], FloatEq(0.25F)) << "joint_vel offset " << i;
    }
    // last_action block: the full 29-wide vector, unchanged.
    const std::size_t action_start = vel_start + kIlNumLegs + kIlNumArmsObs;
    ASSERT_EQ(action_start + kIlNumJoints, kIlObsStepDim);
    for (std::size_t i = 0; i < kIlNumJoints; ++i)
    {
        EXPECT_THAT(obs[action_start + i], FloatEq(last_action[i]));
    }
}

TEST(WalkPolicyIsaacLab, ActionToLegTargetsAppliesScaleAndDefault)
{
    const auto cfg = makeConfig();
    std::array<float, kIlNumLegs> raw{};
    for (std::size_t i = 0; i < kIlNumLegs; ++i) raw[i] = 2.0F;  // +2.0 * 0.25 = +0.5 offset

    const auto targets = ilActionToLegTargets(raw, cfg);
    for (std::size_t i = 0; i < kIlNumLegs; ++i)
    {
        const auto idx      = static_cast<std::size_t>(cfg.leg_action_indices[i]);
        const double expect = cfg.default_joint_pos[idx] + 0.5;
        EXPECT_NEAR(targets[i], expect, 1e-9) << "leg " << i;
    }
}

TEST(WalkPolicyIsaacLab, BuildFullActionPlacesLegsKeepsRestAtZero)
{
    const auto                    cfg = makeConfig();
    std::array<float, kIlNumLegs> raw{};
    for (std::size_t i = 0; i < kIlNumLegs; ++i) raw[i] = static_cast<float>(i) + 1.0F;

    const auto full = ilBuildFullAction(raw, cfg);
    for (std::size_t i = 0; i < kIlNumLegs; ++i)
    {
        const auto idx = static_cast<std::size_t>(cfg.leg_action_indices[i]);
        EXPECT_THAT(full[idx], FloatEq(raw[i])) << "leg slot " << idx;
    }
    // Waist (2, 5, 8) and every arm/other slot not in leg_action_indices stays at zero -- this
    // feeds back as next tick's "last_action" observation term, and an action is a
    // near-zero-centered network output, never an absolute joint angle like default_joint_pos.
    for (std::size_t i = 0; i < kIlNumJoints; ++i)
    {
        const bool is_leg =
            std::find(cfg.leg_action_indices.begin(), cfg.leg_action_indices.end(), static_cast<int>(i)) !=
            cfg.leg_action_indices.end();
        if (!is_leg)
        {
            EXPECT_THAT(full[i], FloatEq(0.0F)) << "slot " << i;
        }
    }
}

TEST(WalkPolicyIsaacLab, HistoryPadsWithFirstStepBeforeFull)
{
    IlObsHistory history;
    std::array<float, kIlObsStepDim> step{};
    step[0] = 42.0F;
    history.push(step);
    EXPECT_FALSE(history.full());

    const auto flat = history.flatten();
    // Every one of the kIlHistLen slots should read back the single pushed step (padding, not
    // zero) -- checked at each slot's own offset for the one marker element.
    for (std::size_t s = 0; s < kIlHistLen; ++s)
    {
        EXPECT_THAT(flat[s * kIlObsStepDim], FloatEq(42.0F)) << "history slot " << s;
    }
}

TEST(WalkPolicyIsaacLab, HistoryEvictsOldestOnceFull)
{
    IlObsHistory history;
    for (std::size_t s = 0; s < kIlHistLen; ++s)
    {
        std::array<float, kIlObsStepDim> step{};
        step[0] = static_cast<float>(s);
        history.push(step);
    }
    ASSERT_TRUE(history.full());

    std::array<float, kIlObsStepDim> new_step{};
    new_step[0] = 999.0F;
    history.push(new_step);

    const auto flat = history.flatten();
    // Oldest (marker 0) is gone; slots now read 1..kIlHistLen-1 then the new 999.
    for (std::size_t s = 0; s + 1 < kIlHistLen; ++s)
    {
        EXPECT_THAT(flat[s * kIlObsStepDim], FloatEq(static_cast<float>(s + 1))) << "slot " << s;
    }
    EXPECT_THAT(flat[(kIlHistLen - 1) * kIlObsStepDim], FloatEq(999.0F));
}

// The default must be NO delay: the reference script's DelayBuffer(5, ...) is a capacity, its
// lag stays at 0. Reading that 5 as the lag added 100 ms of latency and the robot fell.
TEST(WalkPolicyIsaacLab, ActionDelayDefaultsToPassThrough)
{
    IlActionDelay delay;
    EXPECT_EQ(delay.delaySteps(), 0U);
    for (int i = 1; i <= 8; ++i)
    {
        std::array<double, kIlNumLegs> step{};
        step[0]        = static_cast<double>(i);
        const auto out = delay.pushAndPop(step);
        EXPECT_NEAR(out[0], static_cast<double>(i), 1e-9) << "tick " << i;
    }
}

TEST(WalkPolicyIsaacLab, ActionDelayClampsToMax)
{
    IlActionDelay delay(kIlMaxDelaySteps + 10);
    EXPECT_EQ(delay.delaySteps(), kIlMaxDelaySteps);
}

TEST(WalkPolicyIsaacLab, ActionDelayHoldsOldestUntilFullThenReturnsDelayedValue)
{
    constexpr std::size_t kSteps = 3;
    IlActionDelay         delay(kSteps);
    std::array<double, kIlNumLegs> first{};
    first[0] = 1.0;

    // Before the queue has kSteps+1 entries, pushAndPop returns the oldest (first) value --
    // never a hole, never zero.
    auto out = delay.pushAndPop(first);
    EXPECT_NEAR(out[0], 1.0, 1e-9);

    for (int i = 2; i <= static_cast<int>(kSteps); ++i)
    {
        std::array<double, kIlNumLegs> step{};
        step[0] = static_cast<double>(i);
        out     = delay.pushAndPop(step);
        EXPECT_NEAR(out[0], 1.0, 1e-9) << "tick " << i << " should still see the first push";
    }

    // One more push fills the queue past kSteps+1 entries: now the delayed (second) value
    // comes out.
    std::array<double, kIlNumLegs> last{};
    last[0] = 100.0;
    out     = delay.pushAndPop(last);
    EXPECT_NEAR(out[0], 2.0, 1e-9);
}

TEST(WalkPolicyIsaacLab, CommandHeightCrouchesWhileDrivenAndStandsBackUp)
{
    // Never driven: the standing height.
    EXPECT_NEAR(ilCommandHeight(0.8, 0.8, 0.72, -1.0, 1.0, 1.0), 0.8, 1e-12);
    // A command just arrived: straight to the walking height when the step allows it ...
    EXPECT_NEAR(ilCommandHeight(0.8, 0.8, 0.72, 0.0, 1.0, 1.0), 0.72, 1e-12);
    // ... at most max_step per tick otherwise.
    EXPECT_NEAR(ilCommandHeight(0.8, 0.8, 0.72, 0.0, 1.0, 0.05), 0.75, 1e-12);
    // Stopped, but within hold_s: stay crouched, so a pause between Nav2 commands does not
    // put the robot back into the stance it cannot start from.
    EXPECT_NEAR(ilCommandHeight(0.72, 0.8, 0.72, 0.9, 1.0, 1.0), 0.72, 1e-12);
    // Stopped for longer: stand back up.
    EXPECT_NEAR(ilCommandHeight(0.72, 0.8, 0.72, 1.5, 1.0, 1.0), 0.8, 1e-12);
    // walk_height <= 0 disables the switch.
    EXPECT_NEAR(ilCommandHeight(0.8, 0.8, 0.0, 0.0, 1.0, 1.0), 0.8, 1e-12);
}

}  // namespace
}  // namespace g1_motion_service_sim
