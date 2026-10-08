/**
 * @file test_dex5_sim_plugin.cpp
 * @brief G1Dex5SimSystem resolves through the pluginlib lookup controller_manager uses.
 */

#include <gmock/gmock.h>

#include <pluginlib/class_loader.hpp>

#include "hardware_interface/system_interface.hpp"

TEST(G1Dex5SimSystemPluginlib, DiscoversAndInstantiates)
{
    pluginlib::ClassLoader<hardware_interface::SystemInterface> loader(
        "hardware_interface",
        "hardware_interface::SystemInterface");

    ASSERT_TRUE(loader.isClassAvailable("g1_hand_interface/G1Dex5SimSystem"));

    auto instance = loader.createUniqueInstance("g1_hand_interface/G1Dex5SimSystem");
    ASSERT_NE(instance, nullptr);
}
