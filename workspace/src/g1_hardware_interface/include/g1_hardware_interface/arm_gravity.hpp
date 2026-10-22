/**
 * @file arm_gravity.hpp
 * @brief Gravity torques for the arm motors, from the robot's URDF.
 */
#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace g1_hardware_interface
{

/**
 * @brief The torque each arm motor must add to hold its arm still against gravity.
 *
 * The arm motors run a soft PD (kp 25-40) and nothing else: without this, an arm held out sags
 * by a centimetre or more under its own weight and the hand's, and a hand pressed onto a table
 * sags further and stalls. Sent as the motors' feed-forward torque, it leaves the PD only the
 * motion and the contact to deal with.
 *
 * Computed from the whole tree below the root (the pelvis, whose orientation the IMU gives), so
 * each joint carries everything beyond it -- forearm, wrist, the hand and its fingers. Joints
 * whose position is not supplied (fingers, legs) are taken at zero; only their links' masses
 * matter here, and a finger's position moves its centre of mass by millimetres.
 *
 * Plain arithmetic over a flattened tree, no allocation after load(): safe on the RT path.
 */
class ArmGravity
{
public:
    static constexpr std::size_t kNumMotors = 35;

    /**
     * @brief Builds the model from `urdf_xml`, the tree below link `root`.
     * @param motor_of  every joint whose position is read and whose torque is wanted, by name,
     *                  to its LowState motor index (the waist and both arms).
     * @param scaled_roots, mass_scale  every link at or below one of these links has its mass
     *                  multiplied by mass_scale: the simulator's hands weigh 0.170 kg of the
     *                  URDF's 0.777 (patch 003 scales them for the walking policy's sake), and a
     *                  compensation for the real hand would lift the simulated one.
     * @return empty on success, else why it could not be built.
     */
    std::string load(
        const std::string& urdf_xml, const std::string& root,
        const std::unordered_map<std::string, int>& motor_of,
        const std::vector<std::string>& scaled_roots = {}, double mass_scale = 1.0);

    bool loaded() const { return !links_.empty(); }

    /// Total mass below the root, kg: a sanity check for the log.
    double totalMass() const;

    /**
     * @brief Gravity torques at the motors.
     * @param motor_q    LowState joint positions, by motor index.
     * @param quat_wxyz  the root's orientation in the world (w, x, y, z: LowState's IMU order).
     * @param tau        out: for every motor named in load()'s map, the torque that holds it
     *                   still; other entries are left as they are.
     */
    void compute(
        const std::array<double, kNumMotors>& motor_q, const std::array<double, 4>& quat_wxyz,
        std::array<double, kNumMotors>& tau);

private:
    struct Link
    {
        int                   parent{ -1 };  ///< index into links_, -1 for a child of the root
        std::array<double, 9> origin_r{};    ///< joint origin rotation in the parent frame
        std::array<double, 3> origin_p{};    ///< joint origin position in the parent frame
        std::array<double, 3> axis{};        ///< joint axis in the joint frame
        bool                  moving{ false };
        int                   motor{ -1 };  ///< LowState index, -1 when not supplied
        double                mass{ 0.0 };
        std::array<double, 3> com{};  ///< centre of mass in the link frame
    };
    std::vector<Link> links_;  ///< parents before children

    // Per-call scratch, sized in load().
    std::vector<std::array<double, 9>> rot_;
    std::vector<std::array<double, 3>> pos_;
    std::vector<double>                sub_mass_;
    std::vector<std::array<double, 3>> sub_moment_;  ///< sum of mass * world COM, subtree
};

}  // namespace g1_hardware_interface
