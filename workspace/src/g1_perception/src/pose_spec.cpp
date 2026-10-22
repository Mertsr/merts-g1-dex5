#include "g1_perception/pose_spec.hpp"

#include <cmath>
#include <sstream>

namespace g1_perception
{
namespace
{

// Whitespace-separated tokens; a parse that leaves anything over is an error, so a typo such as
// a missing space ("0.1 0.2,0.3") cannot silently drop a value.
std::vector<std::string> tokens(const std::string& text)
{
    std::istringstream       in(text);
    std::vector<std::string> out;
    std::string              t;
    while (in >> t)
    {
        out.push_back(t);
    }
    return out;
}

bool toFloat(const std::string& s, float& out)
{
    try
    {
        std::size_t used = 0;
        out              = std::stof(s, &used);
        return used == s.size() && std::isfinite(out);
    }
    catch (const std::exception&)
    {
        return false;
    }
}

std::optional<Eigen::Isometry3f> poseFromTokens(const std::vector<std::string>& t, std::size_t first,
                                                const std::string& text, std::string& error)
{
    const std::size_t n = t.size() - first;
    if (n != 3 && n != 6)
    {
        error = "'" + text + "': expected x y z or x y z roll pitch yaw";
        return std::nullopt;
    }
    float v[6] = { 0, 0, 0, 0, 0, 0 };
    for (std::size_t i = 0; i < n; ++i)
    {
        if (!toFloat(t[first + i], v[i]))
        {
            error = "'" + text + "': '" + t[first + i] + "' is not a number";
            return std::nullopt;
        }
    }
    return poseFromXyzRpy(v[0], v[1], v[2], v[3], v[4], v[5]);
}

}  // namespace

Eigen::Isometry3f poseFromXyzRpy(float x, float y, float z, float roll, float pitch, float yaw)
{
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    T.linear()          = (Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
                  Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
                  Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()))
                     .toRotationMatrix();
    T.translation() = Eigen::Vector3f(x, y, z);
    return T;
}

std::optional<Eigen::Isometry3f> parsePose(const std::string& text, std::string& error)
{
    return poseFromTokens(tokens(text), 0, text, error);
}

std::optional<FramedPose> parseFramedPose(const std::string& text, std::string& error)
{
    const auto t = tokens(text);
    float      dummy;
    if (t.empty() || toFloat(t[0], dummy))
    {
        error = "'" + text + "': expected a frame name first, then x y z [roll pitch yaw]";
        return std::nullopt;
    }
    const auto pose = poseFromTokens(t, 1, text, error);
    if (!pose)
    {
        return std::nullopt;
    }
    return FramedPose{ t[0], *pose };
}

std::optional<std::vector<Eigen::Isometry3f>> parseSweep(const std::string& text, std::string& error)
{
    const auto t = tokens(text);
    if (t.size() != 4 || (t[0] != "x" && t[0] != "y" && t[0] != "z"))
    {
        error = "'" + text + "': expected <x|y|z> start stop step";
        return std::nullopt;
    }
    float start, stop, step;
    if (!toFloat(t[1], start) || !toFloat(t[2], stop) || !toFloat(t[3], step) || step <= 0.0f || stop < start)
    {
        error = "'" + text + "': start <= stop and step > 0 are required";
        return std::nullopt;
    }
    const long n = static_cast<long>(std::floor((stop - start) / step + 1e-4)) + 1;
    if (n > 10000)
    {
        error = "'" + text + "': " + std::to_string(n) + " steps; that many hypotheses would never finish";
        return std::nullopt;
    }
    const int                      axis = t[0][0] - 'x';
    std::vector<Eigen::Isometry3f> out;
    for (long i = 0; i < n; ++i)
    {
        Eigen::Isometry3f T  = Eigen::Isometry3f::Identity();
        T.translation()[axis] = start + step * static_cast<float>(i);
        out.push_back(T);
    }
    return out;
}

bool Region::contains(const Eigen::Isometry3f& frame_T_pose) const
{
    if (!box.contains(frame_T_pose.translation()))
    {
        return false;
    }
    if (max_angle < 0.0f)
    {
        return true;
    }
    const Eigen::Matrix3f diff = orientation.transpose() * frame_T_pose.linear();
    return Eigen::AngleAxisf(diff).angle() <= max_angle;
}

std::optional<Region> parseRegion(const std::string& text, std::string& error)
{
    const auto t = tokens(text);
    if (t.size() != 7 && t.size() != 8 && t.size() != 11)
    {
        error = "'" + text + "': expected <frame> xmin xmax ymin ymax zmin zmax [max_angle_deg [roll pitch yaw]]";
        return std::nullopt;
    }
    float v[10] = {};
    for (std::size_t i = 1; i < t.size(); ++i)
    {
        if (!toFloat(t[i], v[i - 1]))
        {
            error = "'" + text + "': '" + t[i] + "' is not a number";
            return std::nullopt;
        }
    }
    Region r;
    r.frame = t[0];
    for (int a = 0; a < 3; ++a)
    {
        if (v[2 * a] > v[2 * a + 1])
        {
            error = "'" + text + "': a minimum is above its maximum";
            return std::nullopt;
        }
    }
    r.box = Eigen::AlignedBox3f(Eigen::Vector3f(v[0], v[2], v[4]), Eigen::Vector3f(v[1], v[3], v[5]));
    if (t.size() >= 8)
    {
        r.max_angle = v[6] * static_cast<float>(M_PI) / 180.0f;
    }
    if (t.size() == 11)
    {
        r.orientation = poseFromXyzRpy(0, 0, 0, v[7], v[8], v[9]).linear();
    }
    return r;
}

}  // namespace g1_perception
