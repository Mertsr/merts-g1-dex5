#ifndef G1_HAND_INTERFACE__G1_DEX5_SIM_SYSTEM_HPP_
#define G1_HAND_INTERFACE__G1_DEX5_SIM_SYSTEM_HPP_

/**
 * @file g1_dex5_sim_system.hpp
 * @brief ros2_control SystemInterface for one simulated Dex5 hand. No hardware, no DDS.
 */

#include <array>
#include <atomic>
#include <string>
#include <vector>

#include "g1_hand_interface/dex5_sim_model.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_component_interface_params.hpp"

namespace g1_hand_interface
{

/**
 * @brief One Dex5 hand with no device behind it: the position command is approached smoothly and
 *        reported back as position and velocity, with zero effort.
 *
 * Claims the six active joints only, in the pinned order of kDex5JointSuffixes. A physical backend
 * would implement the same joints and interfaces, so controllers, MoveIt and manipulation code do
 * not change when it replaces this one.
 */
class G1Dex5SimSystem : public hardware_interface::SystemInterface
{
public:
    hardware_interface::CallbackReturn
    on_init(const hardware_interface::HardwareComponentInterfaceParams& params) override;
    hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous) override;
    hardware_interface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State& previous) override;

    std::vector<hardware_interface::StateInterface>   export_state_interfaces() override;
    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

    hardware_interface::return_type
    read(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    hardware_interface::return_type
    write(const rclcpp::Time& time, const rclcpp::Duration& period) override;

private:
    Dex5SimModel model_;
    std::string  side_;

    /// Per joint, in the pinned order.
    std::array<double, kNumDex5Joints> position_command_{};
    std::array<double, kNumDex5Joints> position_state_{};
    std::array<double, kNumDex5Joints> velocity_state_{};
    std::array<double, kNumDex5Joints> effort_state_{};

    /// The model only advances while active, so an inactive hand holds still.
    std::atomic<bool> active_{ false };
};

}  // namespace g1_hand_interface

#endif  // G1_HAND_INTERFACE__G1_DEX5_SIM_SYSTEM_HPP_
