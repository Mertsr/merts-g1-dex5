/**
 * @file walk_policy_isaaclab_session.cpp
 * @brief ONNX Runtime session wrapper for the unitree_sim_isaaclab-sourced walking policy.
 */
#include "g1_motion_service_sim/walk_policy_isaaclab_session.hpp"

#include <chrono>
#include <stdexcept>
#include <vector>

namespace g1_motion_service_sim
{

namespace
{
void checkShape(const std::vector<std::int64_t>& shape, std::size_t expected, const char* what)
{
    if (shape.empty() || static_cast<std::size_t>(shape.back()) != expected)
    {
        throw std::runtime_error(
            std::string("IsaacLab walking policy ") + what + " must be " + std::to_string(expected) +
            " elements wide");
    }
}
}  // namespace

IlWalkPolicySession::IlWalkPolicySession(const std::string& model_path)
  : env_(ORT_LOGGING_LEVEL_WARNING, "g1_walk_policy_isaaclab")
  , memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    options_.SetIntraOpNumThreads(1);
    options_.SetInterOpNumThreads(1);
    options_.SetExecutionMode(ORT_SEQUENTIAL);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options_);

    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
    {
        throw std::runtime_error("IsaacLab walking policy must have exactly one input and one output");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    input_name_  = session_->GetInputNameAllocated(0, allocator).get();
    output_name_ = session_->GetOutputNameAllocated(0, allocator).get();

    checkShape(
        session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape(),
        kIlObsDim,
        "input");
    checkShape(
        session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape(),
        kIlNumLegs,
        "output");

    static constexpr std::array<std::int64_t, 2> kOutputShape{ 1, static_cast<std::int64_t>(kIlNumLegs) };
    output_tensor_ = Ort::Value::CreateTensor<float>(
        memory_info_,
        output_buffer_.data(),
        output_buffer_.size(),
        kOutputShape.data(),
        kOutputShape.size());

    const auto start = std::chrono::steady_clock::now();
    run({});
    warmup_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::array<float, kIlNumLegs> IlWalkPolicySession::run(const std::array<float, kIlObsDim>& observation)
{
    static constexpr std::array<std::int64_t, 2> kInputShape{ 1, static_cast<std::int64_t>(kIlObsDim) };

    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<float*>(observation.data()),
        observation.size(),
        kInputShape.data(),
        kInputShape.size());

    const std::array<const char*, 1> input_names{ input_name_.c_str() };
    const std::array<const char*, 1> output_names{ output_name_.c_str() };

    session_->Run(
        Ort::RunOptions{ nullptr },
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        &output_tensor_,
        1);

    return output_buffer_;
}

}  // namespace g1_motion_service_sim
