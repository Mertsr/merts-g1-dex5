#ifndef G1_HAND_INTERFACE__DEX5_SIM_MODEL_HPP_
#define G1_HAND_INTERFACE__DEX5_SIM_MODEL_HPP_

/**
 * @file dex5_sim_model.hpp
 * @brief Kinematic stand-in for a Dex5 hand's six active joints. No ROS, no DDS.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>

namespace g1_hand_interface
{

/// Active joints per Dex5 hand. The distal joints are URDF mimics and are never claimed.
inline constexpr std::size_t kNumDex5Joints = 6;

/// Joint order, identical for both sides and the same as the URDF's names after the side prefix.
/// This is the ROS-facing contract; a physical backend maps it to its own register order inside
/// the backend.
inline constexpr std::array<const char*, kNumDex5Joints> kDex5JointSuffixes = {
    "thumb_1_joint",  "thumb_2_joint", "index_1_joint",
    "middle_1_joint", "ring_1_joint",  "little_1_joint",
};

/// The URDF's name for joint @p index of the @p side hand, e.g. left_thumb_1_joint.
inline std::string dex5JointName(const std::string& side, std::size_t index)
{
    return side + "_" + kDex5JointSuffixes.at(index);
}

/**
 * @brief First-order, velocity-limited approach toward a position command.
 *
 * Deterministic: the same command sequence and periods always give the same state. Effort is
 * always zero, since this models kinematics only.
 */
class Dex5SimModel
{
public:
    using Array = std::array<double, kNumDex5Joints>;

    /**
     * @param lower Per-joint lower position limit, rad.
     * @param upper Per-joint upper position limit, rad.
     * @param max_velocity Speed cap, rad/s.
     * @param time_constant Seconds for the approach to cover 1/e of the remaining error.
     * @return false, leaving the model unchanged, if any value would make stepping unsafe.
     */
    bool configure(const Array& lower, const Array& upper, double max_velocity, double time_constant)
    {
        if (!std::isfinite(max_velocity) || !std::isfinite(time_constant) || max_velocity <= 0.0 ||
            time_constant <= 0.0)
        {
            return false;
        }
        for (std::size_t i = 0; i < kNumDex5Joints; ++i)
        {
            // Also rules out transposed bounds, for which std::clamp is undefined.
            if (!std::isfinite(lower[i]) || !std::isfinite(upper[i]) || !(lower[i] < upper[i]))
            {
                return false;
            }
        }
        lower_         = lower;
        upper_         = upper;
        max_velocity_  = max_velocity;
        time_constant_ = time_constant;
        reset(position_);
        return true;
    }

    /// Places the joints at @p position, clamped to the limits (0 for a non-finite entry, clamped
    /// too), at rest.
    void reset(const Array& position)
    {
        for (std::size_t i = 0; i < kNumDex5Joints; ++i)
        {
            const double start = std::isfinite(position[i]) ? position[i] : 0.0;
            position_[i]       = std::clamp(start, lower_[i], upper_[i]);
            velocity_[i]       = 0.0;
        }
    }

    /// Zeroes the velocity and keeps the position.
    void stop() { velocity_.fill(0.0); }

    /**
     * @brief Advances by @p dt seconds toward @p command.
     *
     * The command is clamped to the limits and a non-finite entry holds its joint. A period that
     * is not positive and finite changes nothing, and a stalled loop's long period is capped so
     * it cannot widen the step.
     */
    void step(const Array& command, double dt)
    {
        if (!std::isfinite(dt) || dt <= 0.0)
        {
            velocity_.fill(0.0);
            return;
        }
        dt = std::min(dt, kMaxStep);
        for (std::size_t i = 0; i < kNumDex5Joints; ++i)
        {
            const double target = std::isfinite(command[i]) ?
                                      std::clamp(command[i], lower_[i], upper_[i]) :
                                      position_[i];
            const double error  = target - position_[i];
            const double rate   = std::clamp(error / time_constant_, -max_velocity_, max_velocity_);
            double       delta  = rate * dt;
            // Never overshoot the target within one step.
            if (std::abs(delta) > std::abs(error))
            {
                delta = error;
            }
            position_[i] += delta;
            velocity_[i] = delta / dt;
        }
    }

    const Array& position() const { return position_; }
    const Array& velocity() const { return velocity_; }
    const Array& lower() const { return lower_; }
    const Array& upper() const { return upper_; }

private:
    static constexpr double kMaxStep = 0.1;  ///< s

    Array  lower_{};
    Array  upper_{};
    Array  position_{};
    Array  velocity_{};
    double max_velocity_{ 3.0 };
    double time_constant_{ 0.1 };
};

}  // namespace g1_hand_interface

#endif  // G1_HAND_INTERFACE__DEX5_SIM_MODEL_HPP_
