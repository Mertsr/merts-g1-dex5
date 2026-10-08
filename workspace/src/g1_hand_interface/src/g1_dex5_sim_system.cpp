/**
 * @file g1_dex5_sim_system.cpp
 * @brief One simulated Dex5 hand as a ros2_control system. Opens no channels.
 */

#include "g1_hand_interface/g1_dex5_sim_system.hpp"

#include <cstdlib>
#include <optional>
#include <unordered_map>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"

namespace g1_hand_interface
{
namespace
{
/// @p text as a number, or nullopt, so a bad parameter cannot throw out of on_init.
std::optional<double> toDouble(const std::string& text)
{
    char*        end   = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (text.empty() || end == nullptr || *end != '\0')
    {
        return std::nullopt;
    }
    return value;
}

/// Falls back on an absent value; an unparseable one is left to the caller to refuse.
std::optional<double> paramOr(
    const std::unordered_map<std::string, std::string>& params, const std::string& key,
    double fallback)
{
    const auto it = params.find(key);
    return it == params.end() ? std::optional<double>{ fallback } : toDouble(it->second);
}
}  // namespace

hardware_interface::CallbackReturn
G1Dex5SimSystem::on_init(const hardware_interface::HardwareComponentInterfaceParams& params)
{
    if (SystemInterface::on_init(params) != hardware_interface::CallbackReturn::SUCCESS)
    {
        return hardware_interface::CallbackReturn::ERROR;
    }
    const auto& info   = get_hardware_info();
    const auto  logger = rclcpp::get_logger("g1_dex5_sim_system");
    const auto  side   = info.hardware_parameters.find("side");
    if (side == info.hardware_parameters.end() ||
        (side->second != "left" && side->second != "right"))
    {
        RCLCPP_FATAL(logger, "the 'side' parameter must be exactly \"left\" or \"right\"");
        return hardware_interface::CallbackReturn::ERROR;
    }
    side_ = side->second;

    if (info.joints.size() != kNumDex5Joints)
    {
        RCLCPP_FATAL(
            logger,
            "expected %zu active joints for the %s hand but got %zu; the mimic joints are not "
            "claimed",
            kNumDex5Joints,
            side_.c_str(),
            info.joints.size());
        return hardware_interface::CallbackReturn::ERROR;
    }

    Dex5SimModel::Array lower{};
    Dex5SimModel::Array upper{};
    for (std::size_t i = 0; i < kNumDex5Joints; ++i)
    {
        const auto&       joint    = info.joints[i];
        const std::string expected = dex5JointName(side_, i);
        if (joint.name != expected)
        {
            RCLCPP_FATAL(
                logger,
                "joint %zu is '%s' but the Dex5 order needs '%s' there",
                i,
                joint.name.c_str(),
                expected.c_str());
            return hardware_interface::CallbackReturn::ERROR;
        }
        if (joint.command_interfaces.size() != 1 ||
            joint.command_interfaces[0].name != hardware_interface::HW_IF_POSITION)
        {
            RCLCPP_FATAL(logger, "joint '%s' must export only a position command", expected.c_str());
            return hardware_interface::CallbackReturn::ERROR;
        }
        // Required, not defaulted, so a renamed param fails here instead of widening the range.
        const auto lo = joint.parameters.find("min");
        const auto hi = joint.parameters.find("max");
        if (lo == joint.parameters.end() || hi == joint.parameters.end())
        {
            RCLCPP_FATAL(logger, "joint '%s' needs min and max params", expected.c_str());
            return hardware_interface::CallbackReturn::ERROR;
        }
        const std::optional<double> lower_limit = toDouble(lo->second);
        const std::optional<double> upper_limit = toDouble(hi->second);
        if (!lower_limit || !upper_limit)
        {
            RCLCPP_FATAL(logger, "joint '%s' has a non-numeric min or max", expected.c_str());
            return hardware_interface::CallbackReturn::ERROR;
        }
        lower[i] = *lower_limit;
        upper[i] = *upper_limit;
    }

    const auto max_velocity  = paramOr(info.hardware_parameters, "max_joint_velocity_rad_s", 3.0);
    const auto time_constant = paramOr(info.hardware_parameters, "time_constant_s", 0.1);
    if (!max_velocity || !time_constant ||
        !model_.configure(lower, upper, *max_velocity, *time_constant))
    {
        RCLCPP_FATAL(
            logger,
            "the limits must be finite with min < max, and max_joint_velocity_rad_s and "
            "time_constant_s finite and positive");
        return hardware_interface::CallbackReturn::ERROR;
    }

    // Safe resting values before any activation: the model's reset position, which is zero
    // clamped into the limits.
    position_state_   = model_.position();
    position_command_ = model_.position();
    velocity_state_.fill(0.0);
    effort_state_.fill(0.0);
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn G1Dex5SimSystem::on_activate(const rclcpp_lifecycle::State&)
{
    // The command starts at the state, so the first update after activation moves nothing.
    model_.reset(position_state_);
    position_state_   = model_.position();
    position_command_ = model_.position();
    velocity_state_.fill(0.0);
    effort_state_.fill(0.0);
    active_ = true;
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn G1Dex5SimSystem::on_deactivate(const rclcpp_lifecycle::State&)
{
    active_ = false;
    model_.stop();
    velocity_state_.fill(0.0);
    position_command_ = position_state_;
    return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> G1Dex5SimSystem::export_state_interfaces()
{
    std::vector<hardware_interface::StateInterface> interfaces;
    interfaces.reserve(kNumDex5Joints * 3);
    for (std::size_t i = 0; i < kNumDex5Joints; ++i)
    {
        interfaces.emplace_back(
            info_.joints[i].name,
            hardware_interface::HW_IF_POSITION,
            &position_state_[i]);
        interfaces.emplace_back(
            info_.joints[i].name,
            hardware_interface::HW_IF_VELOCITY,
            &velocity_state_[i]);
        interfaces.emplace_back(
            info_.joints[i].name,
            hardware_interface::HW_IF_EFFORT,
            &effort_state_[i]);
    }
    return interfaces;
}

std::vector<hardware_interface::CommandInterface> G1Dex5SimSystem::export_command_interfaces()
{
    std::vector<hardware_interface::CommandInterface> interfaces;
    interfaces.reserve(kNumDex5Joints);
    for (std::size_t i = 0; i < kNumDex5Joints; ++i)
    {
        interfaces.emplace_back(
            info_.joints[i].name,
            hardware_interface::HW_IF_POSITION,
            &position_command_[i]);
    }
    return interfaces;
}

hardware_interface::return_type
G1Dex5SimSystem::read(const rclcpp::Time&, const rclcpp::Duration& period)
{
    if (active_.load())
    {
        // Advances toward the command the controllers wrote on the previous cycle.
        model_.step(position_command_, period.seconds());
        position_state_ = model_.position();
        velocity_state_ = model_.velocity();
    }
    return hardware_interface::return_type::OK;
}

hardware_interface::return_type G1Dex5SimSystem::write(const rclcpp::Time&, const rclcpp::Duration&)
{
    // Nothing to send: the model reads the command interface directly.
    return hardware_interface::return_type::OK;
}

}  // namespace g1_hand_interface

PLUGINLIB_EXPORT_CLASS(g1_hand_interface::G1Dex5SimSystem, hardware_interface::SystemInterface)
