#include "g1_motion_service_sim/walk_policy_isaaclab.hpp"

#include <algorithm>

namespace g1_motion_service_sim
{

std::array<float, kIlObsStepDim> ilAssembleObservationStep(
    const IlPolicyInputs& in, const IlPolicyConfig& cfg,
    const std::array<float, kIlNumJoints>& last_full_action)
{
    std::array<float, kIlObsStepDim> obs{};
    std::size_t                      o = 0;

    for (double v : in.base_ang_vel_body) obs[o++] = static_cast<float>(v);
    for (double v : in.projected_gravity) obs[o++] = static_cast<float>(v);

    obs[o++] = static_cast<float>(in.cmd_vx);
    obs[o++] = static_cast<float>(in.cmd_vy);
    obs[o++] = static_cast<float>(in.cmd_vyaw);
    obs[o++] = static_cast<float>(cfg.command_height);

    // joint_pos: legs then arms, relative to default_joint_pos.
    for (int idx : cfg.leg_action_indices)
        obs[o++] = static_cast<float>(in.joint_pos[static_cast<std::size_t>(idx)] - cfg.default_joint_pos[static_cast<std::size_t>(idx)]);
    for (int idx : cfg.arm_obs_indices)
        obs[o++] = static_cast<float>(in.joint_pos[static_cast<std::size_t>(idx)] - cfg.default_joint_pos[static_cast<std::size_t>(idx)]);

    // joint_vel: same order, relative to default_joint_vel.
    for (int idx : cfg.leg_action_indices)
        obs[o++] = static_cast<float>(in.joint_vel[static_cast<std::size_t>(idx)] - cfg.default_joint_vel[static_cast<std::size_t>(idx)]);
    for (int idx : cfg.arm_obs_indices)
        obs[o++] = static_cast<float>(in.joint_vel[static_cast<std::size_t>(idx)] - cfg.default_joint_vel[static_cast<std::size_t>(idx)]);

    for (float v : last_full_action) obs[o++] = v;

    return obs;  // o == kIlObsStepDim
}

std::array<double, kIlNumLegs> ilActionToLegTargets(
    const std::array<float, kIlNumLegs>& raw_action, const IlPolicyConfig& cfg)
{
    std::array<double, kIlNumLegs> targets{};
    for (std::size_t i = 0; i < kIlNumLegs; ++i)
    {
        const std::size_t idx = static_cast<std::size_t>(cfg.leg_action_indices[i]);
        targets[i]            = cfg.default_joint_pos[idx] + static_cast<double>(raw_action[i]) * cfg.action_scale;
    }
    return targets;
}

std::array<float, kIlNumJoints> ilBuildFullAction(
    const std::array<float, kIlNumLegs>& raw_action, const IlPolicyConfig& cfg)
{
    // This feeds back as next tick's "last_action" observation term (see
    // ilAssembleObservationStep). An action is a near-zero-centered network output, not an
    // absolute joint angle -- IsaacLab's action-manager buffer for an unwritten (non-policy)
    // joint stays at its torch.zeros() initial value for the life of the episode, it is never
    // populated with that joint's default_joint_pos. Filling non-leg slots with
    // default_joint_pos (up to 0.87 rad for a shoulder) was feeding the network a persistent,
    // out-of-distribution signal on ~1/3 of every observation step -- a likely cause of the
    // erratic, fast-twitching joint behaviour seen on hardware-in-sim testing.
    std::array<float, kIlNumJoints> full{};  // zero-initialized: non-leg slots stay 0.
    for (std::size_t i = 0; i < kIlNumLegs; ++i)
        full[static_cast<std::size_t>(cfg.leg_action_indices[i])] = raw_action[i];
    return full;
}

double ilCommandHeight(
    double current, double stand_height, double walk_height, double seconds_since_motion,
    double hold_s, double max_step)
{
    if (walk_height <= 0.0)
        return stand_height;  // switch disabled: the fixed height, as before
    const bool   moving = seconds_since_motion >= 0.0 && seconds_since_motion <= hold_s;
    const double goal   = moving ? walk_height : stand_height;
    if (max_step <= 0.0)
        return goal;
    return std::clamp(goal, current - max_step, current + max_step);
}

}  // namespace g1_motion_service_sim
