/**
 * @file test_wire_contract.cpp
 * @brief The two things about the hand wire format that are easy to get wrong and expensive
 * to discover on hardware: the packed mode byte and the joint order.
 */

#include <gmock/gmock.h>

#include <pluginlib/class_loader.hpp>

#include "g1_hand_interface/g1_hand_system.hpp"
#include "hardware_interface/system_interface.hpp"

using g1_hand_interface::kJointSuffixes;
using g1_hand_interface::kNumHandJoints;
using g1_hand_interface::kStatusFoc;
using g1_hand_interface::kStatusLock;
using g1_hand_interface::packMode;

TEST(HandWireContract, ModePacksIdStatusAndTimeoutIntoTheirOwnFields)
{
    // id in bits 0-3, status in 4-6, timeout in 7. Getting the shift wrong produces a byte
    // the motor still accepts, so nothing complains and the wrong finger moves.
    EXPECT_EQ(packMode(0, kStatusLock, false), 0x00);
    EXPECT_EQ(packMode(3, kStatusFoc, false), 0x13);
    EXPECT_EQ(packMode(5, kStatusFoc, true), 0x95);
    EXPECT_EQ(packMode(5, kStatusLock, true), 0x85);
}

TEST(HandWireContract, EveryMotorCarriesItsOwnIndex)
{
    // id 15 is the broadcast address, so an index that overflowed the nibble would command
    // every motor at once with one finger's target.
    EXPECT_LT(kNumHandJoints, 15U);
    for (std::size_t i = 0; i < kNumHandJoints; ++i)
    {
        EXPECT_EQ(packMode(i, kStatusFoc, false) & 0x0F, i);
    }
}

TEST(HandWireContract, ThereAreSixDrivenJoints)
{
    // The RH56E2 has twelve non-fixed joints and six drivers; the rest are URDF mimics.
    EXPECT_EQ(kNumHandJoints, 6U);
}

TEST(HandWireContract, JointOrderIsThumbThenFingersIndexToLittle)
{
    // The order is positional on the wire and appears in four more places (the xacro, the
    // controllers, the SRDF group and the simulator's responder). g1_moveit_config's drift
    // test compares the files; this pins the table the plugin checks them against.
    EXPECT_THAT(
        kJointSuffixes,
        ::testing::ElementsAre(
            ::testing::StrEq("thumb_1"),
            ::testing::StrEq("thumb_2"),
            ::testing::StrEq("index_1"),
            ::testing::StrEq("middle_1"),
            ::testing::StrEq("ring_1"),
            ::testing::StrEq("little_1")));
}

/**
 * @brief Confirms g1_hand_interface/G1HandSystem is discoverable through pluginlib's
 * ament-index lookup, the same path controller_manager uses, rather than merely compiling.
 */
TEST(G1HandSystemPluginlib, DiscoversAndInstantiates)
{
    pluginlib::ClassLoader<hardware_interface::SystemInterface> loader(
        "hardware_interface",
        "hardware_interface::SystemInterface");

    ASSERT_TRUE(loader.isClassAvailable("g1_hand_interface/G1HandSystem"));

    auto instance = loader.createUniqueInstance("g1_hand_interface/G1HandSystem");
    ASSERT_NE(instance, nullptr);
}
