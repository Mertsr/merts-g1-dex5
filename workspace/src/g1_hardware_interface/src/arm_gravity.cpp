/**
 * @file arm_gravity.cpp
 * @brief Gravity torques for the arm motors, from the robot's URDF.
 */

#include "g1_hardware_interface/arm_gravity.hpp"

#include <urdf/model.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <tuple>
#include <utility>

namespace g1_hardware_interface
{

namespace
{
using Mat3 = std::array<double, 9>;  // row-major
using Vec3 = std::array<double, 3>;

constexpr double kGravity = 9.81;

Mat3 mul(const Mat3& a, const Mat3& b)
{
    Mat3 r{};
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                s +=
                    a[static_cast<std::size_t>(i * 3 + k)] * b[static_cast<std::size_t>(k * 3 + j)];
            }
            r[static_cast<std::size_t>(i * 3 + j)] = s;
        }
    }
    return r;
}

Vec3 mul(const Mat3& a, const Vec3& v)
{
    return { a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
             a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
             a[6] * v[0] + a[7] * v[1] + a[8] * v[2] };
}

Vec3 add(const Vec3& a, const Vec3& b) { return { a[0] + b[0], a[1] + b[1], a[2] + b[2] }; }

Vec3 cross(const Vec3& a, const Vec3& b)
{
    return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}

double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

/// Rotation by `angle` about the unit `axis` (Rodrigues).
Mat3 axisAngle(const Vec3& axis, double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double t = 1.0 - c;
    const double x = axis[0];
    const double y = axis[1];
    const double z = axis[2];
    return { t * x * x + c,     t * x * y - s * z, t * x * z + s * y,
             t * x * y + s * z, t * y * y + c,     t * y * z - s * x,
             t * x * z - s * y, t * y * z + s * x, t * z * z + c };
}

/// Rotation matrix of a quaternion (w, x, y, z), normalised first.
Mat3 fromQuat(double w, double x, double y, double z)
{
    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    if (n < 1e-9)
    {
        return { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    }
    w /= n;
    x /= n;
    y /= n;
    z /= n;
    return { 1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
             2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
             2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y) };
}
}  // namespace

std::string ArmGravity::load(
    const std::string& urdf_xml, const std::string& root,
    const std::unordered_map<std::string, int>& motor_of,
    const std::vector<std::string>& scaled_roots, double mass_scale)
{
    links_.clear();
    urdf::Model model;
    if (!model.initString(urdf_xml))
    {
        return "the robot description does not parse as URDF";
    }
    const auto root_link = model.getLink(root);
    if (!root_link)
    {
        return "the URDF has no link '" + root + "'";
    }

    const auto is_scaled_root = [&](const std::string& name) {
        return std::find(scaled_roots.begin(), scaled_roots.end(), name) != scaled_roots.end();
    };
    // Breadth first from the root: every link after its parent. The flag: inside a scaled
    // subtree.
    std::deque<std::tuple<urdf::LinkConstSharedPtr, int, bool>> queue;
    for (const auto& child : root_link->child_links)
    {
        queue.emplace_back(child, -1, is_scaled_root(child->name));
    }
    std::size_t found = 0;
    while (!queue.empty())
    {
        const auto [link, parent, scaled] = queue.front();
        queue.pop_front();
        const auto& joint = link->parent_joint;
        Link        l;
        l.parent           = parent;
        const auto& origin = joint->parent_to_joint_origin_transform;
        l.origin_p         = { origin.position.x, origin.position.y, origin.position.z };
        l.origin_r =
            fromQuat(origin.rotation.w, origin.rotation.x, origin.rotation.y, origin.rotation.z);
        l.moving = joint->type == urdf::Joint::REVOLUTE || joint->type == urdf::Joint::CONTINUOUS;
        if (l.moving)
        {
            Vec3         a{ joint->axis.x, joint->axis.y, joint->axis.z };
            const double n = std::sqrt(dot(a, a));
            l.axis         = n > 1e-9 ? Vec3{ a[0] / n, a[1] / n, a[2] / n } : Vec3{ 0, 0, 1 };
            const auto it  = motor_of.find(joint->name);
            if (it != motor_of.end())
            {
                l.motor = it->second;
                ++found;
            }
        }
        if (link->inertial)
        {
            l.mass = link->inertial->mass * (scaled ? mass_scale : 1.0);
            l.com  = { link->inertial->origin.position.x,
                       link->inertial->origin.position.y,
                       link->inertial->origin.position.z };
        }
        const int self = static_cast<int>(links_.size());
        links_.push_back(l);
        for (const auto& child : link->child_links)
        {
            queue.emplace_back(child, self, scaled || is_scaled_root(child->name));
        }
    }
    if (found != motor_of.size())
    {
        links_.clear();
        return "only " + std::to_string(found) + " of the " + std::to_string(motor_of.size()) +
               " joints asked for are revolute joints below '" + root + "'";
    }
    rot_.assign(links_.size(), Mat3{});
    pos_.assign(links_.size(), Vec3{});
    sub_mass_.assign(links_.size(), 0.0);
    sub_moment_.assign(links_.size(), Vec3{});
    return "";
}

double ArmGravity::totalMass() const
{
    double m = 0.0;
    for (const auto& l : links_)
    {
        m += l.mass;
    }
    return m;
}

void ArmGravity::compute(
    const std::array<double, kNumMotors>& motor_q, const std::array<double, 4>& quat_wxyz,
    std::array<double, kNumMotors>& tau)
{
    if (links_.empty())
    {
        return;
    }
    // Gravity in the root frame: the world's -z seen from the root, i.e. -g times the world z
    // axis expressed in the root (the third row of the root's world rotation).
    const Mat3 world_r_root = fromQuat(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]);
    const Vec3 g{ -kGravity * world_r_root[6],
                  -kGravity * world_r_root[7],
                  -kGravity * world_r_root[8] };

    // Frames, root frame, parents first.
    const Mat3 identity{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    for (std::size_t i = 0; i < links_.size(); ++i)
    {
        const Link& l   = links_[i];
        const Mat3& p_r = l.parent < 0 ? identity : rot_[static_cast<std::size_t>(l.parent)];
        const Vec3  p_p = l.parent < 0 ? Vec3{} : pos_[static_cast<std::size_t>(l.parent)];
        pos_[i]         = add(p_p, mul(p_r, l.origin_p));
        Mat3 r          = mul(p_r, l.origin_r);
        if (l.moving && l.motor >= 0)
        {
            r = mul(r, axisAngle(l.axis, motor_q[static_cast<std::size_t>(l.motor)]));
        }
        rot_[i]        = r;
        const Vec3 c   = add(pos_[i], mul(r, l.com));
        sub_mass_[i]   = l.mass;
        sub_moment_[i] = { l.mass * c[0], l.mass * c[1], l.mass * c[2] };
    }
    // Subtree sums, children first.
    for (std::size_t i = links_.size(); i-- > 0;)
    {
        const int parent = links_[i].parent;
        if (parent >= 0)
        {
            const auto p = static_cast<std::size_t>(parent);
            sub_mass_[p] += sub_mass_[i];
            sub_moment_[p] = add(sub_moment_[p], sub_moment_[i]);
        }
    }
    // The joint sits at the link frame's origin and turns about the axis (the same in the joint
    // and the link frame). Gravity's generalised force there is axis . ((sum m c - M p) x g);
    // the motor holds it with the opposite.
    for (std::size_t i = 0; i < links_.size(); ++i)
    {
        const Link& l = links_[i];
        if (!l.moving || l.motor < 0)
        {
            continue;
        }
        const Vec3 axis = mul(rot_[i], l.axis);
        const Vec3 arm{ sub_moment_[i][0] - sub_mass_[i] * pos_[i][0],
                        sub_moment_[i][1] - sub_mass_[i] * pos_[i][1],
                        sub_moment_[i][2] - sub_mass_[i] * pos_[i][2] };
        tau[static_cast<std::size_t>(l.motor)] = -dot(axis, cross(arm, g));
    }
}

}  // namespace g1_hardware_interface
