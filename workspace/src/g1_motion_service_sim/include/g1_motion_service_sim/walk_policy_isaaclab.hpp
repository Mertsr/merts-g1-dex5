#ifndef G1_MOTION_SERVICE_SIM__WALK_POLICY_ISAACLAB_HPP_
#define G1_MOTION_SERVICE_SIM__WALK_POLICY_ISAACLAB_HPP_

/**
 * @file walk_policy_isaaclab.hpp
 * @brief Observation assembly and action mapping for the unitree_sim_isaaclab-sourced walking
 *        policy (policy.onnx / policy1.onnx, G129_CFG_WITH_DEX1_WHOLEBODY). Separate contract
 *        from walk_policy.hpp's walker.onnx -- do not merge the two, their observation layouts,
 *        history requirements and action scaling are all different.
 *
 * Contract (reverse-engineered from unitree_sim_isaaclab's DDSRLActionProvider script; index
 * tables and defaults MEASURED against a live IsaacLab environment dump and
 * robots/unitree.py's G129_CFG_WITH_DEX1_WHOLEBODY -- not re-derived from training code we
 * don't have. See g1_locomotion chat log / commit message for full provenance and the three
 * open questions this still carries):
 *
 *   One step    = ang_vel(3) | projected_gravity(3) | command(4, vx/vy/vyaw/height) |
 *                 joint_pos(26: 12 legs then 14 arms, relative to default) |
 *                 joint_vel(26: same order, relative to default) | last_full_action(29)
 *               = 91 elements.
 *   Network in  = 10 stacked steps, oldest-to-newest = 910 elements (IlObsHistory).
 *   Network out = 12 raw actions, in leg_action_indices order (hip_pitch, hip_roll, hip_yaw,
 *                 knee, ankle_pitch, ankle_roll -- each left then right).
 *   Applied cmd = default_joint_pos[leg] + action * action_scale, through IlActionDelay with
 *                 ZERO delay by default. The reference script builds DelayBuffer(5, ...), but
 *                 that 5 is IsaacLab's history CAPACITY, not the lag: DelayBuffer's time lag
 *                 starts at 0 and the script never calls set_time_lag(), so its actions apply
 *                 on the same tick. An earlier version of this file read the 5 as the lag and
 *                 added 100 ms (5 x 20 ms) of actuation latency the policy never saw -- enough
 *                 to make a balancing humanoid chase its own corrections and fall. Waist is
 *                 NEVER commanded by this policy; it stays at default_joint_pos, same as the
 *                 reference script's default_waist_positions.
 *
 * NOT present in this layout, unlike walker.onnx's: no base_lin_vel term at all.
 *
 * Open questions this integration has NOT resolved (see g1_locomotion chat log):
 *   1. DDS<->IsaacLab joint remap (kIlFromDds below) is derived by matching joint NAMES between
 *      g1_joint_index_dds.md and a runtime dump, not confirmed against the actual deployment
 *      script's own remap (if any) in unitree_sim_isaaclab.
 *   2. command_height's exact semantics (confirmed to be a "0.8 base + delta" keyboard-teleop
 *      value, not confirmed to be the thing the policy was actually trained to track).
 *   3. lower_kp/lower_kd (see walk_policy_isaaclab.yaml) come from the SIM articulation's
 *      position-actuator gains, transcribed assuming they transfer 1:1 to /lowcmd's
 *      tau = kp*(q_des-q) + kd*(0-dq) the way walker.onnx's measurably do (see that policy's
 *      own yaml comment) -- NOT independently verified for this policy.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace g1_motion_service_sim
{

inline constexpr std::size_t kIlNumJoints  = 29;  // all_joint_names, IsaacLab runtime order
inline constexpr std::size_t kIlNumLegs    = 12;  // policy's action dimension
inline constexpr std::size_t kIlNumArmsObs = 14;  // observed, never commanded
inline constexpr std::size_t kIlObsStepDim = 91;  // 3+3+4+26+26+29
inline constexpr std::size_t kIlHistLen    = 10;  // fixed: the two shipped ONNX models are [1,910]
inline constexpr std::size_t kIlObsDim     = kIlHistLen * kIlObsStepDim;  // 910
/// Upper bound on IlActionDelay's configurable lag (DelayBuffer(5, ...)'s capacity in the
/// reference script). The DEFAULT lag is 0 -- see the file comment.
inline constexpr std::size_t kIlMaxDelaySteps = 5;

/**
 * @brief IsaacLab runtime joint index -> DDS (Unitree LowState/LowCmd) motor index.
 *
 * Built by matching joint NAMES between g1_joint_index_dds.md's 29DOF table and a live
 * IsaacLab environment dump (user-supplied console output) -- see .hpp file comment,
 * open question 1. Index i here is the IsaacLab position (walk_policy_isaaclab.yaml's
 * joint_names); the value is where that same joint lives in LowState.motor_state[]/
 * LowCmd.motor_cmd[].
 */
inline constexpr std::array<int, kIlNumJoints> kIlFromDds = {
    0,  6,  12, 1,  7,  13, 2,  8,  14, 3,  9,  15, 22, 4,  10,
    16, 23, 5,  11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
};

/**
 * @brief Policy action-output index (0-11) -> DDS motor index.
 *
 * Equivalent to kIlFromDds[leg_action_indices[i]], precomputed: the policy's 12 leg outputs
 * are hip_pitch/hip_roll/hip_yaw/knee/ankle_pitch/ankle_roll, left then right each, which
 * happens to be exactly the DDS left-leg-block/right-leg-block interleave once named -- this
 * array makes that explicit rather than re-deriving it at runtime.
 */
inline constexpr std::array<int, kIlNumLegs> kLegActionToDds = { 0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11 };

/// Static, config-derived quantities: defaults, indices, scale. Gains are NOT here -- the
/// integration reuses MotionServiceSim's existing DDS-order lower_kp_/lower_kd_ (15-wide,
/// legs+waist) via walk_policy_isaaclab.yaml's own lower_kp/lower_kd, already in DDS order,
/// so publishTick()/selectLowerBodyCommand need no changes at all for this policy.
struct IlPolicyConfig
{
    std::array<double, kIlNumJoints> default_joint_pos{};
    std::array<double, kIlNumJoints> default_joint_vel{};
    std::array<int, kIlNumLegs>      leg_action_indices{};  // indices into joint_names (IL order)
    std::array<int, kIlNumArmsObs>   arm_obs_indices{};     // indices into joint_names (IL order)
    double                           action_scale{ 0.25 };
    double                           command_height{ 0.8 };
};

/// Per-tick raw inputs, all 29 joints in IsaacLab (joint_names) order -- the caller remaps
/// from DDS order via kIlFromDds before filling this in.
struct IlPolicyInputs
{
    std::array<double, 3>            base_ang_vel_body{};  // imu gyroscope
    std::array<double, 3>            projected_gravity{};  // world -Z rotated into base frame
    std::array<double, kIlNumJoints> joint_pos{};
    std::array<double, kIlNumJoints> joint_vel{};
    double                           cmd_vx{ 0.0 };
    double                           cmd_vy{ 0.0 };
    double                           cmd_vyaw{ 0.0 };
};

/**
 * @brief Builds one 91-element observation step.
 *
 * joint_pos/joint_vel sub-blocks are ordered leg_action_indices THEN arm_obs_indices (legs
 * first, arms second, each relative to default_joint_pos/vel) -- matches
 * all_obs_indices = action_to_indices + arm_to_all_indices in the reference script.
 *
 * last_full_action is the PREVIOUS tick's full action vector (29, IsaacLab joint_names
 * order): raw (unscaled) policy output at leg_action_indices, default_joint_pos everywhere
 * else. This stack has no external arm-DDS override path, so arms sit at default here
 * exactly as they would in the reference script when no teleop command arrives -- the
 * condition its own base-walking training ran under.
 */
std::array<float, kIlObsStepDim> ilAssembleObservationStep(
    const IlPolicyInputs& in, const IlPolicyConfig& cfg,
    const std::array<float, kIlNumJoints>& last_full_action);

/**
 * @brief Fixed-length history of observation steps, oldest-first on flatten().
 *
 * Mirrors the reference script's CircularBuffer: push evicts the oldest step once full.
 * Before full (first kIlHistLen-1 ticks after a fresh start), flatten() repeats the first
 * available step backward to fill -- matching IsaacLab's CircularBuffer reset behaviour (it
 * seeds every slot with the first sample, not zeros, so the policy never sees an artificial
 * step-zero discontinuity on startup).
 */
class IlObsHistory
{
public:
    void push(const std::array<float, kIlObsStepDim>& step)
    {
        buf_.push_back(step);
        while (buf_.size() > kIlHistLen) buf_.pop_front();
    }

    bool full() const { return buf_.size() >= kIlHistLen; }

    std::array<float, kIlObsDim> flatten() const
    {
        std::array<float, kIlObsDim> out{};
        if (buf_.empty()) return out;
        for (std::size_t i = 0; i < kIlHistLen; ++i)
        {
            const std::size_t pad = kIlHistLen - buf_.size();
            const std::size_t src = (i < pad) ? 0 : (i - pad);
            std::copy(buf_[src].begin(), buf_[src].end(), out.begin() + static_cast<long>(i * kIlObsStepDim));
        }
        return out;
    }

    void reset() { buf_.clear(); }

private:
    std::deque<std::array<float, kIlObsStepDim>> buf_;
};

/**
 * @brief Fixed-length delay queue on the APPLIED 12-dim leg target (DDS-order, post action_scale).
 *
 * pushAndPop(target) enqueues this tick's scaled leg target and returns what was pushed
 * delay_steps ticks ago, or the oldest available value before the queue has filled --
 * never garbage, never a hold at zero that would snap the legs to default_joint_pos.
 * delay_steps == 0 (the default, matching the reference script) returns target unchanged.
 */
class IlActionDelay
{
public:
    explicit IlActionDelay(std::size_t delay_steps = 0)
        : delay_steps_(std::min(delay_steps, kIlMaxDelaySteps))
    {
    }

    std::array<double, kIlNumLegs> pushAndPop(const std::array<double, kIlNumLegs>& target)
    {
        buf_.push_back(target);
        while (buf_.size() > delay_steps_ + 1) buf_.pop_front();
        return buf_.front();
    }

    std::size_t delaySteps() const { return delay_steps_; }

    void reset() { buf_.clear(); }

private:
    std::size_t                                 delay_steps_;
    std::deque<std::array<double, kIlNumLegs>> buf_;
};

/// target[i] = default_joint_pos[leg_action_indices[i]] + raw_action[i] * action_scale, in
/// policy action-output order (i.e. still needs kLegActionToDds to land on DDS motor slots).
std::array<double, kIlNumLegs> ilActionToLegTargets(
    const std::array<float, kIlNumLegs>& raw_action, const IlPolicyConfig& cfg);

/// Builds next tick's 29-wide last_full_action (IsaacLab order): raw_action at
/// leg_action_indices, zero everywhere else -- an action is a near-zero-centered network
/// output, not an absolute joint angle, so an unwritten (non-leg) slot stays at its
/// torch.zeros() initial value rather than default_joint_pos.
std::array<float, kIlNumJoints> ilBuildFullAction(
    const std::array<float, kIlNumLegs>& raw_action, const IlPolicyConfig& cfg);

/**
 * @brief The height command for this tick: walk_height while the robot is being driven (and
 *        for hold_s after the last non-zero command), stand_height otherwise, approached at
 *        most max_step per tick.
 *
 * Why two heights (measured in MuJoCo against policy.onnx on this robot): at 0.8 the policy
 * stands with its knees almost straight (pelvis ~0.79 m). That stance is perfectly still, but
 * from it the policy cannot start a gait -- a 0.2 m/s or 0.6 rad/s command moves the robot a
 * few centimetres and then nothing. Once it is walking it is fine, which is exactly the
 * "freezes on the first move, then walks" symptom. At 0.72 it starts at once, but standing
 * there it shuffles a few cm/s, which is no good next to a table. So: crouch the moment a
 * command arrives, stand back up once the robot has been stopped for hold_s.
 *
 * @param seconds_since_motion seconds since the last non-zero command; negative for "never".
 */
double ilCommandHeight(
    double current, double stand_height, double walk_height, double seconds_since_motion,
    double hold_s, double max_step);

}  // namespace g1_motion_service_sim

#endif  // G1_MOTION_SERVICE_SIM__WALK_POLICY_ISAACLAB_HPP_
