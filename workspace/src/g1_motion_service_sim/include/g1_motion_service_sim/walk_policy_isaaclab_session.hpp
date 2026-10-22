#ifndef G1_MOTION_SERVICE_SIM__WALK_POLICY_ISAACLAB_SESSION_HPP_
#define G1_MOTION_SERVICE_SIM__WALK_POLICY_ISAACLAB_SESSION_HPP_

/**
 * @file walk_policy_isaaclab_session.hpp
 * @brief ONNX Runtime session wrapper for the unitree_sim_isaaclab-sourced walking policy.
 *        Mirrors walk_policy_session.hpp's structure exactly; the only difference is the
 *        910 -> 12 shape contract (kIlObsDim -> kIlNumLegs) instead of 99 -> 29.
 */

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>

#include "g1_motion_service_sim/walk_policy_isaaclab.hpp"

namespace g1_motion_service_sim
{

/**
 * @brief Runs the IsaacLab-sourced walking policy's ONNX graph on the CPU.
 *
 * Single-threaded on purpose, same reasoning as WalkPolicySession: inference runs on the
 * node's own executor thread from a wall timer, and an ORT thread pool would put unmanaged
 * threads inside a node whose whole contract is that one thread touches its state.
 */
class IlWalkPolicySession
{
public:
    /**
     * @param model_path  Filesystem path to policy.onnx (or policy1.onnx).
     * @throws Ort::Exception If the model is missing or malformed.
     * @throws std::runtime_error If the graph's input/output shapes are not 910 -> 12.
     */
    explicit IlWalkPolicySession(const std::string& model_path);

    /**
     * @param observation  Raw 910-element observation (10 stacked 91-element steps).
     * @return The 12 raw policy actions, leg_action_indices order.
     */
    std::array<float, kIlNumLegs> run(const std::array<float, kIlObsDim>& observation);

    double warmupSeconds() const { return warmup_s_; }

private:
    Ort::Env                      env_;
    Ort::SessionOptions           options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo               memory_info_;
    std::string                   input_name_;
    std::string                   output_name_;
    double                        warmup_s_{ 0.0 };

    std::array<float, kIlNumLegs> output_buffer_{};
    Ort::Value                    output_tensor_{ nullptr };
};

}  // namespace g1_motion_service_sim

#endif  // G1_MOTION_SERVICE_SIM__WALK_POLICY_ISAACLAB_SESSION_HPP_
