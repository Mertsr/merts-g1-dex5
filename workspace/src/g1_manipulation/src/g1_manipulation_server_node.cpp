#include "g1_manipulation/g1_manipulation_server_node.hpp"

#include <moveit/robot_state/attached_body.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <string>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <thread>
#include <utility>
#include <vector>

namespace g1_manipulation
{

namespace
{

// A collision box never shrinks below this in any axis. MoveIt treats a zero-extent primitive
// as degenerate and the attach silently does nothing, which then reads as a planner that
// ignored the object.
constexpr double kMinPrimitiveExtent = 0.005;

rclcpp::QoS objectsQos()
{
    return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
}

}  // namespace

bool resolveArm(const std::string& arm, ArmContext& out)
{
    if (arm != "left" && arm != "right")
    {
        return false;
    }
    out.arm_group   = arm + "_arm";
    out.hand_group  = arm + "_hand";
    out.palm_link   = arm + "_hand_palm_link";
    out.grasp_frame = arm + "_hand_grasp_frame";
    out.pinch_frame = arm + "_hand_pinch_frame";
    out.is_left     = arm == "left";
    return true;
}

geometry_msgs::msg::Pose handGoalFromGrasp(const geometry_msgs::msg::Pose& grasp, bool is_left)
{
    if (!is_left)
    {
        return grasp;
    }
    tf2::Transform T;
    tf2::fromMsg(grasp, T);
    tf2::Quaternion flip;
    flip.setRPY(M_PI, 0.0, 0.0);
    T.setRotation(T.getRotation() * flip);
    geometry_msgs::msg::Pose out;
    tf2::toMsg(T, out);
    return out;
}

geometry_msgs::msg::Pose backOff(const geometry_msgs::msg::Pose& pose, double distance)
{
    tf2::Transform T;
    tf2::fromMsg(pose, T);
    const tf2::Vector3 x = tf2::Matrix3x3(T.getRotation()).getColumn(0);
    geometry_msgs::msg::Pose out = pose;
    out.position.x -= distance * x.x();
    out.position.y -= distance * x.y();
    out.position.z -= distance * x.z();
    return out;
}

// The grasp's own name from an attempt label "<object>/grasp/<name> (tilt ..)".
std::string graspNameOf(const std::string& label)
{
    std::string name  = label.substr(0, label.find(" ("));
    const auto  slash = name.rfind("/grasp/");
    return slash == std::string::npos ? name : name.substr(slash + 7);
}

// The hand state a grasp brings for `suffix` ("_open", "_closed"): "<name><suffix>", else the
// name cut back one "_part" at a time, so "side_right_090" finds "side_right_open" or
// "side_open". Empty when the hand has none of them.
std::string shapeFor(
    const std::string& grasp_name, const std::string& suffix, const std::vector<std::string>& shapes)
{
    std::string stem = grasp_name;
    while (!stem.empty())
    {
        if (std::find(shapes.begin(), shapes.end(), stem + suffix) != shapes.end())
        {
            return stem + suffix;
        }
        const auto cut = stem.rfind('_');
        if (cut == std::string::npos)
        {
            break;
        }
        stem.erase(cut);
    }
    return "";
}

// "side_right_090" is for the right hand only, "side_left_090" for the left; a name with neither
// token is for both.
bool graspIsForArm(const std::string& grasp_name, const std::string& arm)
{
    const auto has = [&](const std::string& token) {
        return ("_" + grasp_name + "_").find("_" + token + "_") != std::string::npos;
    };
    if (has("right") && arm != "right")
    {
        return false;
    }
    if (has("left") && arm != "left")
    {
        return false;
    }
    return true;
}

// Fingers about level: the grasp takes the part from beside it, not from above.
bool isSideGrasp(const geometry_msgs::msg::Pose& grasp)
{
    tf2::Quaternion q;
    tf2::fromMsg(grasp.orientation, q);
    return std::abs(tf2::Matrix3x3(q).getColumn(0).z()) < 0.5;
}

// "scoop_right_near": the hand palm up, slid in along the surface under the part's edge, the
// thumb closed down on top (a BSH panel lying on a table: its edge lip stands 27 mm off it).
bool isScoopGrasp(const std::string& grasp_name)
{
    return ("_" + grasp_name + "_").find("_scoop_") != std::string::npos;
}

namespace
{
// The fingers' direction of a pose, with its component along `up` removed; zero if they point
// straight along it.
tf2::Vector3 levelHeading(const geometry_msgs::msg::Quaternion& orientation, const tf2::Vector3& up)
{
    tf2::Quaternion q;
    tf2::fromMsg(orientation, q);
    const tf2::Vector3 x       = tf2::Matrix3x3(q).getColumn(0);
    tf2::Vector3       heading = x - up * x.dot(up);
    if (heading.length() < 1e-6)
    {
        return tf2::Vector3(0.0, 0.0, 0.0);
    }
    return heading.normalized();
}
}  // namespace

geometry_msgs::msg::Pose
slideBack(const geometry_msgs::msg::Pose& pose, double distance, const tf2::Vector3& up)
{
    const tf2::Vector3       heading = levelHeading(pose.orientation, up.normalized());
    geometry_msgs::msg::Pose back    = pose;
    back.position.x -= distance * heading.x();
    back.position.y -= distance * heading.y();
    back.position.z -= distance * heading.z();
    return back;
}

geometry_msgs::msg::Pose
raised(const geometry_msgs::msg::Pose& pose, const tf2::Vector3& up, double distance)
{
    const tf2::Vector3       u   = up.normalized();
    geometry_msgs::msg::Pose out = pose;
    out.position.x += distance * u.x();
    out.position.y += distance * u.y();
    out.position.z += distance * u.z();
    return out;
}

geometry_msgs::msg::Pose
levelScoop(const geometry_msgs::msg::Pose& pose, const tf2::Vector3& up, double pitch)
{
    const tf2::Vector3 u       = up.normalized();
    const tf2::Vector3 heading = levelHeading(pose.orientation, u);
    if (heading.length() < 0.5)
    {
        return pose;
    }
    tf2::Quaternion q;
    tf2::fromMsg(pose.orientation, q);
    const tf2::Vector3 y_was = tf2::Matrix3x3(q).getColumn(1);
    const tf2::Vector3 fx    = heading * std::cos(pitch) - u * std::sin(pitch);
    tf2::Vector3       fy    = heading * std::sin(pitch) + u * std::cos(pitch);
    if (fy.dot(y_was) < 0.0)
    {
        fy = -fy;  // a left hand's goal: turned over about the fingers
    }
    const tf2::Vector3 fz = fx.cross(fy);
    const tf2::Matrix3x3 rot(fx.x(), fy.x(), fz.x(), fx.y(), fy.y(), fz.y(), fx.z(), fy.z(), fz.z());
    tf2::Quaternion level;
    rot.getRotation(level);
    geometry_msgs::msg::Pose out = pose;
    out.orientation              = tf2::toMsg(level);
    return out;
}

double fingerPitch(const geometry_msgs::msg::Quaternion& orientation, const tf2::Vector3& up)
{
    tf2::Quaternion q;
    tf2::fromMsg(orientation, q);
    const double down = -tf2::Matrix3x3(q).getColumn(0).dot(up.normalized());
    return std::asin(std::clamp(down, -1.0, 1.0));
}

std::string GraspVariant::suffix() const
{
    if (std::abs(tilt) < 1e-6 && std::abs(lean) < 1e-6 && !flipped)
    {
        return "";
    }
    char buf[80];
    std::snprintf(buf, sizeof buf, " (tilt %+.0f deg, lean %+.0f deg%s)", tilt * 180.0 / M_PI, lean * 180.0 / M_PI,
                  flipped ? ", turned over" : "");
    return buf;
}

std::vector<GraspVariant> graspVariants(const geometry_msgs::msg::Pose& grasp, double max_tilt, double step,
                                        bool allow_flip, double min_descent)
{
    std::vector<double> tilts{ 0.0 };
    if (step > 1e-6)
    {
        for (double t = step; t <= max_tilt + 1e-9; t += step)
        {
            tilts.push_back(t);
            tilts.push_back(-t);
        }
    }
    tf2::Quaternion base;
    tf2::fromMsg(grasp.orientation, base);
    std::vector<GraspVariant> out;
    for (const bool flipped : allow_flip ? std::vector<bool>{ false, true } : std::vector<bool>{ false })
    {
        tf2::Quaternion over;
        over.setRPY(flipped ? M_PI : 0.0, 0.0, 0.0);
        for (const double t : tilts)
        {
            tf2::Quaternion tilt;
            tilt.setRPY(0.0, t, 0.0);  // about the grasp's own y: the closing axis
            const tf2::Quaternion q = (base * over * tilt).normalized();
            if (tf2::Matrix3x3(q).getColumn(0).z() > -min_descent)
            {
                continue;  // fingers level or pointing up: not coming down onto the part
            }
            GraspVariant v;
            v.tilt          = t;
            v.flipped       = flipped;
            v.pose          = grasp;
            v.pose.orientation = tf2::toMsg(q);
            out.push_back(v);
        }
    }
    return out;
}

std::optional<geometry_msgs::msg::Pose> findHypothesis(
    const vision_msgs::msg::Detection3D& detection, const std::string& class_id)
{
    for (const auto& r : detection.results)
    {
        if (r.hypothesis.class_id == class_id)
        {
            return r.pose.pose;
        }
    }
    return std::nullopt;
}

std::optional<std::pair<std::string, geometry_msgs::msg::Pose>> findHypothesisPrefix(
    const vision_msgs::msg::Detection3D& detection, const std::string& prefix)
{
    for (const auto& r : detection.results)
    {
        if (r.hypothesis.class_id.rfind(prefix, 0) == 0)
        {
            return std::make_pair(r.hypothesis.class_id, r.pose.pose);
        }
    }
    return std::nullopt;
}

G1ManipulationServer::G1ManipulationServer(const rclcpp::NodeOptions& options)
  : rclcpp::Node("g1_manipulation_server", options)
{
    object_timeout_s_         = declare_parameter<double>("object_timeout_ms", 1000.0) / 1000.0;
    approach_height_m_        = declare_parameter<double>("approach_height_m", 0.22);
    grasp_height_above_top_m_ = declare_parameter<double>("grasp_height_above_top_m", 0.010);
    lift_height_m_            = declare_parameter<double>("lift_height_m", 0.15);
    // How far a released object may be from where it was aimed before the place is a failure.
    place_tolerance_m_ = declare_parameter<double>("place_tolerance_m", 0.08);
    // Well under the joint limits' own 0.8 rad/s cap. Arm motion disturbs a standing humanoid
    // measurably, and slowing the whole path is preferred over clamping joints, which would
    // bend the path itself.
    velocity_scaling_ = declare_parameter<double>("velocity_scaling", 0.3);
    planning_time_s_  = declare_parameter<double>("planning_time_s", 5.0);

    // How the hand is held at the grasp, for the RIGHT hand; the left mirrors its roll.
    //
    // WHERE the hand grips is not here: it is the {side}_hand_grasp_frame link in the URDF,
    // which is a property of the hand's geometry rather than of this task. Goals are given for
    // that frame, so nothing in this file adds an offset to a palm pose.
    //
    // The fingers close toward the palm's +y, so it is the ROLL that turns the closing axis
    // downward for a grasp off a table, and the pitch that would be wrong.
    grasp_rpy_ = declare_parameter<std::vector<double>>("grasp_rpy", { -M_PI_2, 0.0, 0.0 });
    // For objects that publish their own grasp ("<object>/grasp/<name>" on /objects): the hand
    // goes to a point this far back along the grasp's approach axis, then straight in.
    // 0.20, not less: the RH56E2's straight fingers reach 0.11 m past its grasp point, so a
    // shorter stand-off puts the fingertips at the part's top already at the pregrasp, and the
    // transit there drags them across it.
    grasp_approach_m_    = declare_parameter<double>("grasp_approach_m", 0.20);
    // Re-aiming before the descent follows a part this far at most; further, it was knocked.
    reaim_max_shift_m_   = declare_parameter<double>("reaim_max_shift_m", 0.04);
    locate_timeout_s_    = declare_parameter<double>("locate_timeout_s", 3.0);
    linear_min_fraction_ = declare_parameter<double>("linear_min_fraction", 0.95);
    // A published grasp is tried turned about its closing axis by up to this much, in these
    // steps, and with the hand turned over if allowed; only tilts whose fingers still point down
    // by at least grasp_min_descent (the z of the approach direction) are kept.
    grasp_tilt_max_rad_  = declare_parameter<double>("grasp_tilt_max_deg", 15.0) * M_PI / 180.0;
    grasp_tilt_step_rad_ = declare_parameter<double>("grasp_tilt_step_deg", 15.0) * M_PI / 180.0;
    grasp_allow_flip_    = declare_parameter<bool>("grasp_allow_flip", false);
    // ... and leaned about the pinched edge (the grasp's z) by up to this much.
    grasp_lean_max_rad_  = declare_parameter<double>("grasp_lean_max_deg", 60.0) * M_PI / 180.0;
    // After clearing the octomap to get a stuck arm moving, how long the sensors get to refill
    // it before re-planning (the updaters run at up to 5 Hz).
    octomap_refill_s_    = declare_parameter<double>("octomap_refill_s", 1.0);
    // The octomap lives in the planning frame, the pelvis, so a robot that walked to the table
    // (go_to_stance.py) carries every voxel it saw on the way along with it: the table smeared
    // where it is not, and missing where it is. Every pick therefore starts from a cleared map
    // and waits this long for the LiDAR and the depth camera to fill it from where it stands.
    pick_octomap_refill_s_ = declare_parameter<double>("pick_octomap_refill_s", 2.5);
    // A scoop also puts the table itself into the planning scene, as a box, independently of
    // the octomap: its top scoop_surface_margin_m below the table (so the part lying on it
    // and the hand's collision hulls in their scoop pose -- coarser than the hand itself, and
    // within 2 cm of the table at the start of the slide -- are clear of it), the table's top taken
    // scoop_surface_below_grasp_m under the published scoop grasp (measured: the panel's
    // scoop_right_near grasp frame stands 0.095 m above the table it lies on), reaching
    // scoop_surface_front_m toward the robot from the grasp (the conveyor_flat table's edge is
    // 0.15 in front of it) and 0.6 m beyond, 1.2 m wide. Removed when the pick ends.
    scoop_surface_below_grasp_m_ = declare_parameter<double>("scoop_surface_below_grasp_m", 0.095);
    scoop_surface_margin_m_      = declare_parameter<double>("scoop_surface_margin_m", 0.02);
    scoop_surface_front_m_       = declare_parameter<double>("scoop_surface_front_m", 0.12);
    scoop_surface_thickness_m_   = declare_parameter<double>("scoop_surface_thickness_m", 0.05);
    // Whether the box is added at all. Off: with the octomap working the table is in it, from
    // the very pick that clears and refills it, and the box only duplicated it -- it was added
    // when the octomap was silently empty (moveit_ros_perception was not installed).
    scoop_table_box_             = declare_parameter<bool>("scoop_table_box", false);
    // Before the clamping hand reaches for the other half, the octomap is cleared and refilled
    // for this long. What it held from before the lift is stale by then -- the part's own voxels
    // where it lay, whatever the gripping arm was in front of -- and the part, attached now, is
    // cut out of every new cloud.
    clamp_octomap_refill_s_      = declare_parameter<double>("clamp_octomap_refill_s", 1.5);
    grasp_min_descent_   = declare_parameter<double>("grasp_min_descent", 0.25);
    grasp_max_plans_     = static_cast<int>(declare_parameter<int64_t>("grasp_max_plans", 6));
    side_grasp_min_forward_ = declare_parameter<double>("side_grasp_min_forward", 0.5);
    side_grasp_ik_timeout_s_ = declare_parameter<double>("side_grasp_ik_timeout_s", 0.5);
    // Each placement of a scoop is screened with IK this long per pose: there are up to 25 of
    // them, most of the ones that fail fail on the first pose.
    scoop_variant_ik_timeout_s_ = declare_parameter<double>("scoop_variant_ik_timeout_s", 0.1);
    // A scoop grasp starts this far back from the grasp along the fingers' heading, at the
    // grasp's height (the fingertips 1-2 cm in front of the edge), and comes down to there from
    // this far above it.
    scoop_slide_m_ = declare_parameter<double>("scoop_slide_m", 0.11);
    scoop_hover_m_ = declare_parameter<double>("scoop_hover_m", 0.08);
    // ... and is aimed this much higher than published. At the published height the hand
    // slides in 6 mm (fingertips) and 9 mm (palm) above the table -- measured in MuJoCo against
    // the conveyor_flat panel with the arm exactly on its target -- so a centimetre of arm sag
    // or of error in the part's estimated height puts the back of the hand on the table, every
    // time. The panel's near edge stands proud of the table, and the full two-hand pick, carry
    // and place in MuJoCo behaved the same from +4 mm to +45 mm above the published grasp
    // (roll, sag and carry within a few tenths, placed within 0.7 deg). 25 mm sits in the
    // middle: the hand slides in about 3 cm clear of the table, with 2 cm to spare above.
    scoop_raise_m_ = declare_parameter<double>("scoop_raise_m", 0.025);
    // A scooped part hangs from one edge and sags along its length (a 0.6 m BSH panel: 9-20
    // degrees, the fingers' 1 Nm cannot hold it level). With scoop_two_hand the other hand then
    // comes under it, palm up, and the two lift and carry it together (MuJoCo, sim gains: sag
    // 2-3 degrees, swing under 0.2). Where it goes: scoop_support_offset from the gripping hand's
    // lifted grasp frame, along its fingers' level heading, sideways toward the part's middle,
    // and up (m; measured on the panel: under its other half, where the fingers then sit 1.7 cm
    // above the panel's levelled underside, pressing it up level). It comes in from
    // scoop_support_back_m in front of that and scoop_support_drop_m below, then rises into
    // place; the two then lift scoop_support_lift_m more together. A place lowers them together
    // to support_release_height_m above the target, takes the supporting hand out, and sets the
    // part down with the gripping hand alone.
    scoop_two_hand_       = declare_parameter<bool>("scoop_two_hand", true);
    scoop_support_offset_ = declare_parameter<std::vector<double>>(
        "scoop_support_offset",
        std::vector<double>{ -0.02, 0.314, 0.017 });
    scoop_support_drop_m_     = declare_parameter<double>("scoop_support_drop_m", 0.08);
    scoop_support_back_m_     = declare_parameter<double>("scoop_support_back_m", 0.12);
    scoop_support_lift_m_     = declare_parameter<double>("scoop_support_lift_m", 0.05);
    support_release_height_m_ = declare_parameter<double>("support_release_height_m", 0.025);
    if (scoop_support_offset_.size() != 3)
    {
        throw std::runtime_error("scoop_support_offset needs exactly 3 entries");
    }
    // How the other hand holds it. "clamp" (default): a second scoop in the air, on the same
    // edge's other half -- palm up, fingers in under the edge, thumb down on top, closed -- so the
    // part is clamped by both hands' fingers, not just lying on one. In MuJoCo (sim gains) a 2 N
    // push at the panel's end moved it 5 mm / 4 degrees and it came back to within 3 mm / 2.5;
    // on a flat hand under it ("support") a push of that kind lifted it off the hand and left it
    // 44 degrees out. Where it goes: scoop_clamp_offset from the gripping hand's lifted grasp
    // frame (heading, sideways toward the middle, up; measured on the panel: the edge's other
    // half, 5 cm lower because the part sags until the second hand has it), fingers pitched
    // scoop_clamp_pitch_deg down; it comes in along its fingers from scoop_slide_m in front.
    // Holds within +-1-2 cm of that pose. "none": one hand only.
    scoop_second_hand_  = declare_parameter<std::string>("scoop_second_hand", "clamp");
    scoop_two_hand_     = scoop_two_hand_ && scoop_second_hand_ != "none";
    scoop_clamp_offset_ = declare_parameter<std::vector<double>>(
        "scoop_clamp_offset",
        std::vector<double>{ 0.002, 0.352, -0.050 });
    scoop_clamp_pitch_rad_ =
        declare_parameter<double>("scoop_clamp_pitch_deg", 20.0) * M_PI / 180.0;
    // Where the clamping hand is planned to before it comes down to its start (the start itself
    // is low over the table, inside its voxels): (back along its fingers, up) pairs, tried in
    // turn until one is reachable -- which one is depends on where the robot stands.
    scoop_clamp_pre_m_ = declare_parameter<std::vector<double>>(
        "scoop_clamp_approaches",
        std::vector<double>{ 0.12, 0.06, 0.0, 0.08 });
    // The scoop, level with the world: its fingers this far below the horizontal whatever the
    // part's estimated tilt (published scoops are 25 degrees), and up taken from gravity_frame,
    // not the pelvis. Once the hand is down at the table, a hand more than
    // scoop_pitch_tolerance_deg off that (the robot leaning as it reaches) stops the slide: flat,
    // the hand's base meets the table's edge.
    gravity_frame_   = declare_parameter<std::string>("gravity_frame", "odom");
    scoop_pitch_rad_ = declare_parameter<double>("scoop_pitch_deg", 25.0) * M_PI / 180.0;
    scoop_pitch_tolerance_rad_ =
        declare_parameter<double>("scoop_pitch_tolerance_deg", 8.0) * M_PI / 180.0;
    // Where the robot stands is not exact. A scoop is also tried with the gripping hand turned
    // about the vertical (deg, counter-clockwise seen from above, for the right hand; mirrored
    // for the left) from square to the edge, and the clamping hand turned too, in the order
    // given, until both hands reach: in MuJoCo that widens the stances the two-hand scoop works
    // from (see the README). Only the turns that held there: the right hand turned clockwise
    // grips the edge so that the panel twists and the clamp misses it (sag 13-15 degrees, a
    // 5 N push lost it). Not shifted along the edge either: the panel then hangs from a
    // different point, sags by a different angle, and the clamp's offset no longer finds it.
    {
        const auto to_rad = [](const std::vector<double>& deg) {
            std::vector<double> rad;
            for (const double d : deg)
            {
                rad.push_back(d * M_PI / 180.0);
            }
            return rad;
        };
        scoop_turns_rad_       = to_rad(declare_parameter<std::vector<double>>(
            "scoop_turns_deg",
            std::vector<double>{ 0.0, 10.0, 20.0 }));
        scoop_clamp_turns_rad_ = to_rad(declare_parameter<std::vector<double>>(
            "scoop_clamp_turns_deg",
            std::vector<double>{ 0.0, 10.0, 20.0 }));
    }
    // Where the right hand's scoop point is best seen from, in the pelvis frame (x ahead, y
    // left): the middle of the stances it works from. Only for telling the user where to stand.
    scoop_stance_aim_ = declare_parameter<std::vector<double>>(
        "scoop_stance_aim",
        std::vector<double>{ 0.46, -0.16 });
    if (scoop_clamp_pre_m_.empty() || scoop_clamp_pre_m_.size() % 2 != 0 ||
        scoop_turns_rad_.empty() || scoop_clamp_turns_rad_.empty() || scoop_stance_aim_.size() != 2)
    {
        throw std::runtime_error(
            "scoop_clamp_approaches needs (back, up) pairs, scoop_turns_deg and "
            "scoop_clamp_turns_deg at least one entry, scoop_stance_aim two");
    }
    if (scoop_clamp_offset_.size() != 3 ||
        (scoop_second_hand_ != "clamp" && scoop_second_hand_ != "support" &&
         scoop_second_hand_ != "none"))
    {
        throw std::runtime_error(
            "scoop_clamp_offset needs 3 entries; scoop_second_hand is clamp, support or none");
    }

    objects_sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
        "/objects",
        objectsQos(),
        [this](const vision_msgs::msg::Detection3DArray::ConstSharedPtr& msg) { onObjects(msg); });

    tf_buffer_   = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    get_scene_   = create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
    apply_scene_ = create_client<moveit_msgs::srv::ApplyPlanningScene>("/apply_planning_scene");
    // Reach checks go to move_group's IK service rather than an IK solver in this process: the
    // solver plugin is configured, loaded and proven where the planning happens, and an
    // in-process copy failed to instantiate here ("No kinematics solver instantiated").
    compute_ik_ = create_client<moveit_msgs::srv::GetPositionIK>("/compute_ik");
    check_validity_ = create_client<moveit_msgs::srv::GetStateValidity>("/check_state_validity");
    clear_octomap_ = create_client<std_srvs::srv::Empty>("/clear_octomap");
}

bool G1ManipulationServer::allowHandContact(
    const ArmContext& arm, const std::vector<std::string>& touchables, bool allowed,
    bool include_links)
{
    // The links that unavoidably enter occupied space during a grasp: the hand itself, and
    // the wrist that carries it. Taken from the robot model rather than listed, so a renamed
    // link is a build-time problem rather than a silently ineffective exemption.
    MoveGroup* hand = groupFor(arm.hand_group);
    if (hand == nullptr)
    {
        return false;
    }
    std::vector<std::string> links =
        hand->getRobotModel()->getJointModelGroup(arm.hand_group)->getLinkModelNames();
    const std::string side = arm.is_left ? "left" : "right";
    links.push_back(arm.palm_link);
    links.push_back(side + "_wrist_pitch_link");
    links.push_back(side + "_wrist_yaw_link");
    // All THREE wrist joints, matching the touch_links the pick attaches the object with. Roll
    // is easy to miss but is the one that reaches during a grasp, so omitting it leaves the
    // exemption silently incomplete.
    links.push_back(side + "_wrist_roll_link");

    // The current matrix has to be read first: ApplyPlanningScene replaces the whole ACM
    // rather than merging into it, so sending only our entries would drop every
    // self-collision rule the SRDF set up.
    auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    request->components.components =
        moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;
    auto future = get_scene_->async_send_request(request);
    // Waited on WITHOUT spinning: the executor already owns this node, and spinning it from
    // here would re-enter the executor from inside its own callback.
    if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
    {
        RCLCPP_ERROR(get_logger(), "/get_planning_scene did not answer");
        return false;
    }

    moveit_msgs::msg::AllowedCollisionMatrix acm = future.get()->scene.allowed_collision_matrix;

    // A name the matrix has never seen has to be added as a full row and column first,
    // otherwise the indices below run off the end.
    const auto ensure_entry = [&acm](const std::string& name) {
        if (std::find(acm.entry_names.begin(), acm.entry_names.end(), name) != acm.entry_names.end())
        {
            return;
        }
        acm.entry_names.push_back(name);
        for (moveit_msgs::msg::AllowedCollisionEntry& row : acm.entry_values)
        {
            row.enabled.push_back(false);
        }
        moveit_msgs::msg::AllowedCollisionEntry row;
        row.enabled.assign(acm.entry_names.size(), false);
        acm.entry_values.push_back(row);
    };
    const auto index_of = [&acm](const std::string& name) {
        return static_cast<std::size_t>(
            std::find(acm.entry_names.begin(), acm.entry_names.end(), name) -
            acm.entry_names.begin());
    };

    for (const std::string& touchable : touchables)
    {
        ensure_entry(touchable);
    }
    // The exempted things also have to be allowed to touch EACH OTHER, not just the hand.
    // Once the object is attached it stops being a world object and starts moving with the
    // arm, and lifting it out of a surface drags it through the octomap voxels of that
    // surface -- which is a collision between two things the hand is already allowed to
    // touch, and would otherwise fail the lift with the grasp already made.
    for (std::size_t i = 0; i < touchables.size(); ++i)
    {
        for (std::size_t j = i + 1; j < touchables.size(); ++j)
        {
            const std::size_t a            = index_of(touchables[i]);
            const std::size_t b            = index_of(touchables[j]);
            acm.entry_values[a].enabled[b] = allowed;
            acm.entry_values[b].enabled[a] = allowed;
        }
    }
    for (const std::string& touchable : include_links ? touchables : std::vector<std::string>{})
    {
        const std::size_t other = index_of(touchable);
        for (const std::string& link : links)
        {
            const auto it = std::find(acm.entry_names.begin(), acm.entry_names.end(), link);
            if (it == acm.entry_names.end())
            {
                continue;
            }
            const auto index = static_cast<std::size_t>(it - acm.entry_names.begin());
            acm.entry_values[index].enabled[other] = allowed;
            acm.entry_values[other].enabled[index] = allowed;
        }
    }

    auto apply           = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
    apply->scene.is_diff = true;
    apply->scene.allowed_collision_matrix = acm;
    auto applied                          = apply_scene_->async_send_request(apply);
    if (applied.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
    {
        RCLCPP_ERROR(get_logger(), "/apply_planning_scene did not answer");
        return false;
    }
    RCLCPP_INFO(
        get_logger(),
        "%s contact between the %s hand and %zu object(s)",
        allowed ? "allowing" : "restoring",
        side.c_str(),
        touchables.size());
    return applied.get()->success;
}

std::optional<geometry_msgs::msg::Pose> G1ManipulationServer::toPlanningFrame(
    const geometry_msgs::msg::Pose& pose, const std::string& frame_id)
{
    if (frame_id.empty() || frame_id == planning_frame_)
    {
        return pose;
    }

    geometry_msgs::msg::PoseStamped in;
    in.header.frame_id = frame_id;
    // Deliberately the latest available rather than a stamp: the source's own stamp can be a
    // few sample periods old, and tf2 would then either extrapolate or refuse. The robot
    // stands still to manipulate, so latest is the right reading.
    in.header.stamp = rclcpp::Time(0);
    in.pose         = pose;

    try
    {
        return tf_buffer_->transform(in, planning_frame_, std::chrono::milliseconds(500)).pose;
    }
    catch (const tf2::TransformException& e)
    {
        RCLCPP_ERROR(
            get_logger(),
            "cannot transform '%s' into the planning frame '%s': %s",
            frame_id.c_str(),
            planning_frame_.c_str(),
            e.what());
        return std::nullopt;
    }
}

void G1ManipulationServer::initialize()
{
    if (grasp_rpy_.size() != 3)
    {
        throw std::runtime_error("grasp_rpy needs exactly 3 entries");
    }

    // both_arms included: the mission tree uses it to tuck both arms in one motion before
    // walking, and a group this node does not build is a goal it rejects outright. It is only
    // ever driven to named postures here -- pose goals for it would need the subgroup IK map,
    // which is why g1.srdf deliberately gives it no kinematics entry.
    for (const std::string& name :
         { "left_arm", "right_arm", "both_arms", "left_hand", "right_hand" })
    {
        auto group = std::make_shared<MoveGroup>(shared_from_this(), name);
        group->setMaxVelocityScalingFactor(velocity_scaling_);
        group->setMaxAccelerationScalingFactor(velocity_scaling_);
        group->setPlanningTime(planning_time_s_);
        groups_.emplace(name, group);
    }
    planning_frame_ = groups_.at("left_arm")->getPlanningFrame();

    // Servers come up only once the groups are usable. Advertising first would accept a goal
    // this node cannot yet act on, and the caller would see a timeout rather than a refusal.
    pick_server_ = rclcpp_action::create_server<Pick>(
        this,
        "~/pick",
        [](const rclcpp_action::GoalUUID&, const std::shared_ptr<const Pick::Goal>&) {
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](const std::shared_ptr<GoalHandle<Pick>>&) {
            return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](const std::shared_ptr<GoalHandle<Pick>>& handle) {
            std::thread{ [this, handle] { executePick(handle); } }.detach();
        });

    place_server_ = rclcpp_action::create_server<Place>(
        this,
        "~/place",
        [](const rclcpp_action::GoalUUID&, const std::shared_ptr<const Place::Goal>&) {
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](const std::shared_ptr<GoalHandle<Place>>&) {
            return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](const std::shared_ptr<GoalHandle<Place>>& handle) {
            std::thread{ [this, handle] { executePlace(handle); } }.detach();
        });

    posture_server_ = rclcpp_action::create_server<SetArmPosture>(
        this,
        "~/set_arm_posture",
        [](const rclcpp_action::GoalUUID&, const std::shared_ptr<const SetArmPosture::Goal>&) {
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](const std::shared_ptr<GoalHandle<SetArmPosture>>&) {
            return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](const std::shared_ptr<GoalHandle<SetArmPosture>>& handle) {
            std::thread{ [this, handle] { executeSetArmPosture(handle); } }.detach();
        });

    RCLCPP_INFO(
        get_logger(),
        "Manipulation skills up, planning in '%s'. Executing needs the arm and hands already "
        "acquired; this node never takes control authority itself.",
        planning_frame_.c_str());
}

void G1ManipulationServer::onObjects(const vision_msgs::msg::Detection3DArray::ConstSharedPtr& msg)
{
    std::lock_guard<std::mutex> lock(objects_mutex_);
    objects_ = *msg;
}

std::optional<vision_msgs::msg::Detection3D>
G1ManipulationServer::lookUpObject(const std::string& object_id)
{
    vision_msgs::msg::Detection3DArray snapshot;
    {
        std::lock_guard<std::mutex> lock(objects_mutex_);
        snapshot = objects_;
    }

    // Age is judged here rather than at the source, because only the thing about to commit an
    // arm to a grasp knows how old a pose is too old. g1_object_pose_source deliberately
    // forwards the sample's own stamp so this check means something.
    const double age = (now() - rclcpp::Time(snapshot.header.stamp)).seconds();
    if (age > object_timeout_s_)
    {
        RCLCPP_ERROR(
            get_logger(),
            "No usable object poses: newest %.2f s old (limit %.2f). Is g1_object_pose_source "
            "active, and is its source publishing?",
            age,
            object_timeout_s_);
        return std::nullopt;
    }

    for (const vision_msgs::msg::Detection3D& detection : snapshot.detections)
    {
        if (!detection.results.empty() && detection.results.front().hypothesis.class_id == object_id)
        {
            return detection;
        }
    }
    // Several of one part are reported as <id>_1, <id>_2, ... (the tracker's `instances`): asked
    // for <id>, take the one nearest the pelvis, which is the one the arm is likeliest to reach.
    {
        std::optional<vision_msgs::msg::Detection3D> nearest;
        double                                       best = 1e9;
        const std::string                            prefix = object_id + "_";
        for (const vision_msgs::msg::Detection3D& detection : snapshot.detections)
        {
            if (detection.results.empty())
            {
                continue;
            }
            const std::string& id = detection.results.front().hypothesis.class_id;
            if (id.size() <= prefix.size() || id.rfind(prefix, 0) != 0 ||
                id.find_first_not_of("0123456789", prefix.size()) != std::string::npos)
            {
                continue;
            }
            const std::string frame = detection.header.frame_id.empty() ? snapshot.header.frame_id
                                                                        : detection.header.frame_id;
            const auto pose = toPlanningFrame(detection.results.front().pose.pose, frame);
            if (!pose)
            {
                continue;
            }
            const double d = std::hypot(pose->position.x, pose->position.y);
            if (d < best)
            {
                best    = d;
                nearest = detection;
            }
        }
        if (nearest)
        {
            RCLCPP_INFO(get_logger(), "'%s': taking %s, %.2f m from the pelvis", object_id.c_str(),
                        nearest->results.front().hypothesis.class_id.c_str(), best);
            return nearest;
        }
    }
    // Fresh, but without it: a measuring source (the depth tracker) reports only what it is
    // tracking right now. Say what IS there, which is usually the answer.
    std::string seen;
    for (const vision_msgs::msg::Detection3D& detection : snapshot.detections)
    {
        if (!detection.results.empty())
        {
            seen += (seen.empty() ? "" : ", ") + detection.results.front().hypothesis.class_id;
        }
    }
    RCLCPP_ERROR(
        get_logger(),
        "No object called '%s' is being reported right now (reported: %s). With the depth "
        "tracker, the part has to be in the head camera's view and tracked.",
        object_id.c_str(),
        seen.empty() ? "nothing" : seen.c_str());
    return std::nullopt;
}

moveit_msgs::msg::CollisionObject G1ManipulationServer::publishCollisionObject(
    const vision_msgs::msg::Detection3D& detection,
    const geometry_msgs::msg::Pose&      in_planning_frame)
{
    moveit_msgs::msg::CollisionObject object;
    object.id              = detection.results.front().hypothesis.class_id;
    object.header.frame_id = planning_frame_;

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    // A box for every object, whatever its real shape. The planner only needs a conservative
    // volume to route around, and the bounding box the pose source reports is exactly that.
    primitive.dimensions = { std::max(detection.bbox.size.x, kMinPrimitiveExtent),
                             std::max(detection.bbox.size.y, kMinPrimitiveExtent),
                             std::max(detection.bbox.size.z, kMinPrimitiveExtent) };

    object.primitives.push_back(primitive);
    object.primitive_poses.push_back(in_planning_frame);
    object.operation = moveit_msgs::msg::CollisionObject::ADD;

    // ADD on an id that already exists replaces it, so a re-plan against a moved object does
    // not need a remove first.
    planning_scene_.applyCollisionObjects({ object });
    // Returned so the attach can rebuild the same geometry: by then the world copy has been
    // removed, and an attached body has to carry its own shape.
    return object;
}

geometry_msgs::msg::Pose G1ManipulationServer::graspFrameGoal(
    const geometry_msgs::msg::Pose& object_pose, double object_height_m, const ArmContext& arm) const
{
    // Horizontally the grasp frame goes straight to the object: that frame is defined as the
    // point the hand closes on, so putting it at the object IS the grasp.
    geometry_msgs::msg::Pose goal;
    goal.position = object_pose.position;

    // Vertically it does NOT. Measured from the object's TOP FACE, not its centre.
    //
    // The roll points the closing axis at the floor and the fingertips sit about 24 mm beyond
    // the grasp frame along it, so this offset decides where the fingers end up. Aiming at the
    // centre would put them inside the octomap's 20 mm padding around the surface, closing
    // through it rather than around the object's upper half.
    const double top = object_pose.position.z + 0.5 * object_height_m;
    goal.position.z  = top + grasp_height_above_top_m_;

    // Only the orientation is a choice, and it mirrors: the two hands close in opposite
    // directions, so the roll that points the closing axis at the floor flips sign.
    tf2::Quaternion rotation;
    rotation.setRPY((arm.is_left ? -1.0 : 1.0) * grasp_rpy_[0], grasp_rpy_[1], grasp_rpy_[2]);
    goal.orientation = tf2::toMsg(rotation);
    return goal;
}

bool G1ManipulationServer::moveTo(
    MoveGroup& group, const geometry_msgs::msg::Pose& pose, const std::string& link,
    const std::string& what)
{
    group.setStartStateToCurrentState();
    // Named explicitly rather than relying on the group's default tip: the goal is for the
    // grasp frame, which hangs off the palm and is not what the group ends at.
    group.setPoseTarget(pose, link);

    MoveGroup::Plan plan;
    if (!planWithRecovery(group, plan, what))
    {
        return false;
    }
    const auto executed = group.execute(plan);
    if (executed != moveit::core::MoveItErrorCode::SUCCESS)
    {
        RCLCPP_ERROR(
            get_logger(),
            "%s: execution failed (%d). The usual cause is the arm not being acquired -- run "
            "g1_bringup's activate_arm first.",
            what.c_str(),
            executed.val);
        return false;
    }
    return true;
}

bool G1ManipulationServer::moveToNamed(MoveGroup& group, const std::string& named_target)
{
    group.setStartStateToCurrentState();
    if (!group.setNamedTarget(named_target))
    {
        RCLCPP_ERROR(
            get_logger(),
            "'%s' is not a named pose of group '%s'",
            named_target.c_str(),
            group.getName().c_str());
        return false;
    }

    // Plan then execute, NOT move(). They are not equivalent: move() runs through MoveIt's
    // PlanExecution, which re-checks the whole remaining path against every planning-scene
    // update and aborts the moment one invalidates it. Against a live octomap fed by a chest
    // camera staring at the arm, the arm's own freshly integrated voxels invalidate it
    // constantly, so a plan that started fine can die partway through against itself.
    //
    // execute() checks nothing during the motion, matching every other move in this file: Pick
    // and Place plan and execute for the pregrasp, descent and lift, all fully collision-checked
    // at PLAN time. This only drops a during-flight recheck that, on this stack, reports the
    // robot's own arm.
    MoveGroup::Plan plan;
    if (!planWithRecovery(group, plan, "'" + named_target + "'"))
    {
        return false;
    }
    return group.execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
}

bool G1ManipulationServer::moveLinear(
    MoveGroup& group, const geometry_msgs::msg::Pose& pose, const std::string& link,
    const std::string& what)
{
    group.setStartStateToCurrentState();
    // computeCartesianPath moves the group's end-effector link; the goal is for the grasp
    // frame, so it stands in as the end effector for this one call.
    const std::string previous = group.getEndEffectorLink();
    group.setEndEffectorLink(link);
    moveit_msgs::msg::RobotTrajectory trajectory;
    const double fraction = group.computeCartesianPath({ pose }, 0.005, 0.0, trajectory, true);
    group.setEndEffectorLink(previous);
    if (fraction < linear_min_fraction_)
    {
        // Which it is -- something in the way, or the arm itself (a joint limit, or an IK jump
        // between the waypoints) -- is the same call without the collision check.
        group.setEndEffectorLink(link);
        moveit_msgs::msg::RobotTrajectory unchecked;
        const double free = group.computeCartesianPath({ pose }, 0.005, 0.0, unchecked, false);
        group.setEndEffectorLink(previous);
        RCLCPP_ERROR(
            get_logger(),
            "%s: only %.0f%% of the straight line is reachable without collision (%.0f%% "
            "ignoring collisions: %s)",
            what.c_str(),
            100.0 * fraction,
            100.0 * free,
            free > fraction + 1e-3 ? "something is in the way" : "the arm cannot follow the line");
        return false;
    }
    // A Cartesian path comes back with timing for full speed; slow it to the skills' scaling,
    // the same reason every other move here is scaled.
    robot_trajectory::RobotTrajectory timed(group.getRobotModel(), group.getName());
    timed.setRobotTrajectoryMsg(*group.getCurrentState(), trajectory);
    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    if (!totg.computeTimeStamps(timed, velocity_scaling_, velocity_scaling_))
    {
        RCLCPP_ERROR(get_logger(), "%s: could not time the straight-line path", what.c_str());
        return false;
    }
    timed.getRobotTrajectoryMsg(trajectory);
    const auto executed = group.execute(trajectory);
    if (executed != moveit::core::MoveItErrorCode::SUCCESS)
    {
        RCLCPP_ERROR(get_logger(), "%s: execution failed (%d)", what.c_str(), executed.val);
        return false;
    }
    return true;
}

G1ManipulationServer::MoveGroup* G1ManipulationServer::groupFor(const std::string& name)
{
    const auto it = groups_.find(name);
    return it == groups_.end() ? nullptr : it->second.get();
}

void G1ManipulationServer::usePinchFrame(ArmContext& arm)
{
    // Published grasps are pinches, aimed at where the thumb meets the index finger. Falls back
    // to grasp_frame on a description without the pinch frame.
    if (groupFor(arm.arm_group)->getRobotModel()->hasLinkModel(arm.pinch_frame))
    {
        arm.grasp_frame = arm.pinch_frame;
    }
}

std::optional<bool> G1ManipulationServer::hasIk(
    const std::string& group, const std::string& link, const geometry_msgs::msg::Pose& pose,
    std::map<std::string, double>* joints, double timeout_s, bool avoid_collisions)
{
    ik_broken_code_ = 0;
    if (!compute_ik_->wait_for_service(std::chrono::seconds(2)))
    {
        return std::nullopt;
    }
    auto request                          = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
    request->ik_request.group_name        = group;
    request->ik_request.ik_link_name      = link;
    request->ik_request.robot_state.is_diff = true;  // seeded from the current state
    // Reach only by default; the planner checks collisions.
    request->ik_request.avoid_collisions             = avoid_collisions;
    request->ik_request.pose_stamped.header.frame_id = planning_frame_;
    request->ik_request.pose_stamped.pose = pose;
    request->ik_request.timeout                      = rclcpp::Duration::from_seconds(timeout_s);
    auto future = compute_ik_->async_send_request(request);
    // Waited on without spinning, like the planning-scene calls: the executor owns this node.
    if (future.wait_for(std::chrono::duration<double>(2.0 + timeout_s)) != std::future_status::ready)
    {
        return std::nullopt;
    }
    const auto response = future.get();
    const int  code     = response->error_code.val;
    if (code == moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
    {
        if (joints)
        {
            // Only the group's own joints: the rest of the state is what it already is.
            const auto& names = groupFor(group)->getActiveJoints();
            const auto& js    = response->solution.joint_state;
            joints->clear();
            for (std::size_t i = 0; i < js.name.size() && i < js.position.size(); ++i)
            {
                if (std::find(names.begin(), names.end(), js.name[i]) != names.end())
                {
                    (*joints)[js.name[i]] = js.position[i];
                }
            }
        }
        return true;
    }
    if (code == moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION ||
        code == moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT)
    {
        return false;
    }
    // Anything else is move_group unable to run IK at all -- typically no solver loaded for the
    // group -- which must not be reported as "out of reach".
    RCLCPP_ERROR(get_logger(), "/compute_ik for '%s' failed with MoveIt error code %d", group.c_str(), code);
    ik_broken_code_ = code;
    return std::nullopt;
}

void G1ManipulationServer::refreshUp()
{
    // The planning frame is the pelvis, which leans and pitches as the robot balances a reaching
    // arm; the table is level with the world. Up is the gravity frame's z seen from the pelvis.
    try
    {
        const auto t = tf_buffer_->lookupTransform(
            planning_frame_,
            gravity_frame_,
            tf2::TimePointZero,
            tf2::durationFromSec(0.5));
        tf2::Quaternion q;
        tf2::fromMsg(t.transform.rotation, q);
        up_ = tf2::quatRotate(q, tf2::Vector3(0.0, 0.0, 1.0)).normalized();
        RCLCPP_INFO(
            get_logger(),
            "the pelvis leans %.1f deg from the vertical (%s)",
            std::acos(std::clamp(up_.z(), -1.0, 1.0)) * 180.0 / M_PI,
            gravity_frame_.c_str());
    }
    catch (const tf2::TransformException& e)
    {
        up_ = tf2::Vector3(0.0, 0.0, 1.0);
        RCLCPP_WARN(
            get_logger(),
            "no TF from %s to %s (%s): taking the pelvis as level",
            gravity_frame_.c_str(),
            planning_frame_.c_str(),
            e.what());
    }
}

G1ManipulationServer::ClampPoses G1ManipulationServer::clampPoses(
    const geometry_msgs::msg::Pose& lifted, double gripping_turn, bool clamping_is_left,
    double pre_back, double pre_up, double clamping_turn) const
{
    // Relative to the part's edge, not to the gripping hand, which may have come in turned by
    // `gripping_turn`: the edge's inward normal is that hand's level heading turned back. From
    // the gripping hand's lifted grasp point: along that normal, along the edge toward the
    // clamping hand's own side (its left for the left hand), and up. Palm up, the fingers along
    // the normal turned by `clamping_turn` and pitched scoop_clamp_pitch down.
    tf2::Quaternion lq;
    tf2::fromMsg(lifted.orientation, lq);
    tf2::Vector3 heading      = tf2::Matrix3x3(lq).getColumn(0);
    heading                   = (heading - up_ * heading.dot(up_)).normalized();
    const tf2::Vector3 normal = tf2::quatRotate(tf2::Quaternion(up_, -gripping_turn), heading);
    tf2::Vector3       side   = up_.cross(normal);  // along the edge, to the gripping hand's left
    if (!clamping_is_left)
    {
        side = -side;
    }
    const tf2::Vector3 at = tf2::Vector3(lifted.position.x, lifted.position.y, lifted.position.z) +
                            normal * scoop_clamp_offset_[0] + side * scoop_clamp_offset_[1] +
                            up_ * scoop_clamp_offset_[2];
    const tf2::Vector3   fingers = tf2::quatRotate(tf2::Quaternion(up_, clamping_turn), normal);
    const tf2::Vector3   across  = fingers.cross(up_);
    const tf2::Matrix3x3 flat(
        fingers.x(),
        up_.x(),
        across.x(),
        fingers.y(),
        up_.y(),
        across.y(),
        fingers.z(),
        up_.z(),
        across.z());
    tf2::Quaternion fq;
    flat.getRotation(fq);
    geometry_msgs::msg::Pose palm_up;
    palm_up.position.x  = at.x();
    palm_up.position.y  = at.y();
    palm_up.position.z  = at.z();
    palm_up.orientation = tf2::toMsg(fq);
    palm_up             = levelScoop(palm_up, up_, scoop_clamp_pitch_rad_);

    ClampPoses out;
    out.at       = handGoalFromGrasp(palm_up, clamping_is_left);
    out.in_front = slideBack(out.at, scoop_slide_m_, up_);
    out.pre      = raised(slideBack(out.in_front, pre_back, up_), up_, pre_up);
    out.lifted   = raised(out.at, up_, scoop_support_lift_m_);
    return out;
}

void G1ManipulationServer::refreshOctomap(double wait_s)
{
    if (!clear_octomap_->wait_for_service(std::chrono::seconds(1)))
    {
        RCLCPP_WARN(
            get_logger(),
            "/clear_octomap is not there: planning against the octomap as it is");
        return;
    }
    auto future =
        clear_octomap_->async_send_request(std::make_shared<std_srvs::srv::Empty::Request>());
    future.wait_for(std::chrono::seconds(2));
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(wait_s * 1000)));
    RCLCPP_INFO(
        get_logger(),
        "octomap cleared and refilled for %.1f s from where the robot stands",
        wait_s);
}

std::string G1ManipulationServer::addSurfaceSlab(
    const std::string& object_id, const geometry_msgs::msg::Pose& grasp)
{
    tf2::Quaternion q;
    tf2::fromMsg(grasp.orientation, q);
    tf2::Vector3 heading = tf2::Matrix3x3(q).getColumn(0);
    heading              = heading - up_ * heading.dot(up_);
    if (heading.length() < 1e-3)
    {
        return "";
    }
    heading.normalize();
    const tf2::Vector3 side      = up_.cross(heading);
    const double       thickness = scoop_surface_thickness_m_;
    const double       back = scoop_surface_front_m_, ahead = 0.6;
    // The slab's middle: along the heading halfway between its near and far edges, and its
    // thickness below its top.
    const tf2::Vector3 g(grasp.position.x, grasp.position.y, grasp.position.z);
    const tf2::Vector3 top_at_grasp =
        g - up_ * (scoop_surface_below_grasp_m_ + scoop_surface_margin_m_);
    const tf2::Vector3 centre =
        top_at_grasp + heading * (0.5 * (ahead - back)) - up_ * (0.5 * thickness);
    const tf2::Matrix3x3 rot(
        heading.x(),
        side.x(),
        up_.x(),
        heading.y(),
        side.y(),
        up_.y(),
        heading.z(),
        side.z(),
        up_.z());
    tf2::Quaternion rq;
    rot.getRotation(rq);

    moveit_msgs::msg::CollisionObject slab;
    slab.id              = object_id + "/surface";
    slab.header.frame_id = planning_frame_;
    shape_msgs::msg::SolidPrimitive box;
    box.type       = shape_msgs::msg::SolidPrimitive::BOX;
    box.dimensions = { back + ahead, 1.2, thickness };
    slab.primitives.push_back(box);
    geometry_msgs::msg::Pose pose;
    pose.position.x  = centre.x();
    pose.position.y  = centre.y();
    pose.position.z  = centre.z();
    pose.orientation = tf2::toMsg(rq);
    slab.primitive_poses.push_back(pose);
    slab.operation = moveit_msgs::msg::CollisionObject::ADD;
    planning_scene_.applyCollisionObject(slab);
    RCLCPP_INFO(
        get_logger(),
        "pick %s: the table as a box, its top at pelvis z %.3f, from %.2f m in front of the grasp",
        object_id.c_str(),
        top_at_grasp.z(),
        back);
    return slab.id;
}

std::vector<std::string> G1ManipulationServer::collisionsAt(
    const std::string& group, const std::map<std::string, double>& joints)
{
    std::vector<std::string> pairs;
    if (joints.empty() || !check_validity_->wait_for_service(std::chrono::seconds(1)))
    {
        return pairs;
    }
    auto request                         = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
    request->group_name                  = group;
    request->robot_state.is_diff         = true;
    for (const auto& [name, value] : joints)
    {
        request->robot_state.joint_state.name.push_back(name);
        request->robot_state.joint_state.position.push_back(value);
    }
    auto future = check_validity_->async_send_request(request);
    if (future.wait_for(std::chrono::seconds(3)) != std::future_status::ready)
    {
        return pairs;
    }
    for (const auto& c : future.get()->contacts)
    {
        const std::string pair = c.contact_body_1 + " / " + c.contact_body_2;
        if (std::find(pairs.begin(), pairs.end(), pair) == pairs.end())
        {
            pairs.push_back(pair);
        }
    }
    return pairs;
}

std::size_t G1ManipulationServer::clearStaleWorldObjects()
{
    // A world object stays where it was put, in the planning frame -- the pelvis -- so a part's
    // box left behind by an earlier pick that stopped before attaching it walks along with the
    // robot afterwards and stands where no part is: a "ghost" panel, solid to the planner,
    // blocking hands that reach past it. Nothing else in this stack adds world objects, and every
    // pick re-adds what it needs from the current detection, so all of them go. Attached
    // objects (a part still in a hand) are not world objects and are not touched.
    const std::vector<std::string> names = planning_scene_.getKnownObjectNames();
    if (!names.empty())
    {
        planning_scene_.removeCollisionObjects(names);
        std::string list;
        for (const std::string& n : names)
        {
            list += (list.empty() ? "" : ", ") + n;
        }
        RCLCPP_INFO(get_logger(), "removed %zu stale world object(s) from the scene: %s", names.size(),
                    list.c_str());
    }
    return names.size();
}

void G1ManipulationServer::removeSurfaceSlab(const std::string& id)
{
    if (id.empty())
    {
        return;
    }
    moveit_msgs::msg::CollisionObject remove;
    remove.id        = id;
    remove.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    planning_scene_.applyCollisionObject(remove);
}

bool G1ManipulationServer::planWithRecovery(MoveGroup& group, MoveGroup::Plan& plan, const std::string& what)
{
    auto planned = group.plan(plan);
    if (planned == moveit::core::MoveItErrorCode::SUCCESS)
    {
        return true;
    }
    // The usual cause after a pick that got part-way: the hand is now where the octomap still
    // has voxels from before it got there (the self-filter only stops NEW points of the robot
    // being mapped), so the start state is in collision and every plan fails, including the one
    // to move away. Clearing the map and letting the sensors refill it -- without the hand, which
    // they filter out -- is the fix. Waited for, so the retry does not plan in an empty world.
    RCLCPP_WARN(get_logger(), "%s: planning failed (%d); clearing the octomap and retrying once", what.c_str(),
                planned.val);
    if (clear_octomap_->wait_for_service(std::chrono::seconds(1)))
    {
        auto future = clear_octomap_->async_send_request(std::make_shared<std_srvs::srv::Empty::Request>());
        future.wait_for(std::chrono::seconds(2));
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(octomap_refill_s_ * 1000)));
    }
    group.setStartStateToCurrentState();
    planned = group.plan(plan);
    if (planned != moveit::core::MoveItErrorCode::SUCCESS)
    {
        RCLCPP_ERROR(get_logger(), "%s: planning failed again (%d)", what.c_str(), planned.val);
        return false;
    }
    return true;
}

bool G1ManipulationServer::moveToJoints(MoveGroup& group, const std::map<std::string, double>& joints,
                                        const std::string& what)
{
    group.setStartStateToCurrentState();
    if (!group.setJointValueTarget(joints))
    {
        RCLCPP_ERROR(get_logger(), "%s: joint target rejected", what.c_str());
        return false;
    }
    MoveGroup::Plan plan;
    if (!planWithRecovery(group, plan, what))
    {
        return false;
    }
    if (group.execute(plan) != moveit::core::MoveItErrorCode::SUCCESS)
    {
        RCLCPP_ERROR(get_logger(), "%s: execution failed", what.c_str());
        return false;
    }
    return true;
}

bool G1ManipulationServer::moveBothTo(
    const ArmContext& a, const geometry_msgs::msg::Pose& pose_a, const ArmContext& b,
    const geometry_msgs::msg::Pose& pose_b, const std::string& what)
{
    MoveGroup* both = groupFor("both_arms");
    if (both == nullptr)
    {
        RCLCPP_ERROR(get_logger(), "%s: no both_arms group", what.c_str());
        return false;
    }
    std::map<std::string, double> ja, jb;
    const bool                    reach_a =
        hasIk(a.arm_group, a.grasp_frame, pose_a, &ja, side_grasp_ik_timeout_s_).value_or(false);
    const bool reach_b =
        reach_a &&
        hasIk(b.arm_group, b.grasp_frame, pose_b, &jb, side_grasp_ik_timeout_s_).value_or(false);
    if (!reach_b)
    {
        const ArmContext&               out = reach_a ? b : a;
        const geometry_msgs::msg::Pose& p   = reach_a ? pose_b : pose_a;
        RCLCPP_ERROR(
            get_logger(),
            "%s: the %s hand cannot reach pelvis (%.2f, %.2f, %.2f): stand nearer the target",
            what.c_str(),
            out.is_left ? "left" : "right",
            p.position.x,
            p.position.y,
            p.position.z);
        return false;
    }
    const moveit::core::RobotStatePtr start = both->getCurrentState();
    if (start == nullptr)
    {
        RCLCPP_ERROR(get_logger(), "%s: no current state", what.c_str());
        return false;
    }
    const moveit::core::JointModelGroup* const jmg = start->getJointModelGroup("both_arms");
    if (jmg == nullptr)
    {
        RCLCPP_ERROR(get_logger(), "%s: no both_arms group in the model", what.c_str());
        return false;
    }
    std::vector<double> q0;
    start->copyJointGroupPositions(jmg, q0);
    const std::vector<std::string>& names = jmg->getVariableNames();
    std::vector<double>             q1    = q0;
    for (std::size_t i = 0; i < names.size() && i < q1.size(); ++i)
    {
        if (const auto it = ja.find(names[i]); it != ja.end())
        {
            q1[i] = it->second;
        }
        else if (const auto jt = jb.find(names[i]); jt != jb.end())
        {
            q1[i] = jt->second;
        }
    }
    robot_trajectory::RobotTrajectory trajectory(start->getRobotModel(), "both_arms");
    constexpr int                     kSteps = 20;
    for (int k = 0; k <= kSteps; ++k)
    {
        moveit::core::RobotState state(*start);
        std::vector<double>      q(q0.size());
        for (std::size_t i = 0; i < q.size(); ++i)
        {
            q[i] = q0[i] + (q1[i] - q0[i]) * static_cast<double>(k) / kSteps;
        }
        state.setJointGroupPositions(jmg, q);
        state.update();
        trajectory.addSuffixWayPoint(state, 0.0);
    }
    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    if (!totg.computeTimeStamps(trajectory, velocity_scaling_, velocity_scaling_))
    {
        RCLCPP_ERROR(get_logger(), "%s: could not time the two-arm move", what.c_str());
        return false;
    }
    moveit_msgs::msg::RobotTrajectory msg;
    trajectory.getRobotTrajectoryMsg(msg);
    const auto executed = both->execute(msg);
    if (executed != moveit::core::MoveItErrorCode::SUCCESS)
    {
        RCLCPP_ERROR(get_logger(), "%s: execution failed (%d)", what.c_str(), executed.val);
        return false;
    }
    return true;
}

void G1ManipulationServer::executePick(const std::shared_ptr<GoalHandle<Pick>>& goal_handle)
{
    const auto goal     = goal_handle->get_goal();
    auto       result   = std::make_shared<Pick::Result>();
    auto       feedback = std::make_shared<Pick::Feedback>();

    if (goal->arm != "left" && goal->arm != "right" && goal->arm != "auto")
    {
        result->success = false;
        result->message = "arm must be 'left', 'right' or 'auto', got '" + goal->arm + "'";
        goal_handle->abort(result);
        return;
    }
    // Located before the arm is chosen: with arm "auto" the grasp point decides it. Nothing has
    // moved yet, so a failure here needs no clean-up.
    const auto abort_locating = [&](const std::string& why) {
        result->success = false;
        result->message = std::string(Pick::Feedback::PHASE_LOCATING) + ": " + why;
        goal_handle->abort(result);
    };

    feedback->phase = Pick::Feedback::PHASE_LOCATING;
    goal_handle->publish_feedback(feedback);
    // Waited for, briefly: a tracked part can be between frames, or still being searched for
    // the moment the robot has turned to it.
    auto detection = lookUpObject(goal->object_id);
    const auto locate_deadline = now() + rclcpp::Duration::from_seconds(locate_timeout_s_);
    while (!detection && now() < locate_deadline && rclcpp::ok())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        detection = lookUpObject(goal->object_id);
    }
    if (!detection)
    {
        abort_locating("no fresh pose for '" + goal->object_id + "' within " +
                       std::to_string(locate_timeout_s_) + " s");
        return;
    }
    // The instance actually being picked ("panel_2" when asked for "panel"): what the grasps,
    // the collision object and the attachment are named after from here on.
    const std::string object_id = detection->results.front().hypothesis.class_id;
    // /objects is in odom and the planner works in pelvis; the two differ by wherever the
    // robot is standing, so every measured pose goes through TF before it is planned against.
    const std::string object_frame =
        detection->header.frame_id.empty() ? objects_.header.frame_id : detection->header.frame_id;
    const auto object_pose = toPlanningFrame(detection->results.front().pose.pose, object_frame);
    if (!object_pose)
    {
        abort_locating("could not transform the object pose");
        return;
    }
    // Where the grasp is aimed at, updated if the part is seen to move before the descent.
    geometry_msgs::msg::Pose aimed_object_pose = *object_pose;
    // An object that publishes its own grasps ("<object>/grasp/<name>", from g1_perception's
    // tracker) is taken at one of them, straight in along the grasp's approach axis. Anything
    // else gets the top grasp graspFrameGoal() builds from its bounding box.
    struct Candidate
    {
        std::string                             name;
        std::optional<geometry_msgs::msg::Pose> pose;  ///< planning frame; empty = bounding box
    };
    std::vector<Candidate> candidates;
    for (const auto& r : detection->results)
    {
        if (r.hypothesis.class_id.rfind(object_id + "/grasp/", 0) != 0)
        {
            continue;
        }
        const auto pose = toPlanningFrame(r.pose.pose, object_frame);
        if (!pose)
        {
            abort_locating("could not transform " + r.hypothesis.class_id);
            return;
        }
        candidates.push_back({ r.hypothesis.class_id, pose });
    }
    refreshUp();
    clearStaleWorldObjects();
    refreshOctomap(pick_octomap_refill_s_);

    // The table under a scooped part, as a box (see scoop_surface_*): taken from the first
    // scoop grasp, levelled. Removed again however the pick ends.
    std::string surface_id;
    for (const Candidate& c : candidates)
    {
        if (scoop_table_box_ && c.pose && isScoopGrasp(graspNameOf(c.name)))
        {
            surface_id = addSurfaceSlab(object_id, levelScoop(*c.pose, up_, scoop_pitch_rad_));
            break;
        }
    }
    struct SlabGuard
    {
        G1ManipulationServer* self;
        const std::string*    id;
        ~SlabGuard() { self->removeSurfaceSlab(*id); }
    } slab_guard{ this, &surface_id };
    // What the hand (and the part, once attached) may touch while it is in contact: the octomap,
    // the part, and the table's box -- the part is lifted off the table it overlaps.
    std::vector<std::string> held_touch{ "<octomap>", object_id };
    if (!surface_id.empty())
    {
        held_touch.push_back(surface_id);
    }

    // Fingers pointing most AWAY from the robot first: a part's grasps usually come as mirror
    // pairs tipped toward either end of it, and the one leaning away is the one an arm reaching
    // out from the pelvis can take. The others are still tried, in that order.
    const auto forwardness = [](const geometry_msgs::msg::Pose& pose) {
        tf2::Quaternion q;
        tf2::fromMsg(pose.orientation, q);
        return tf2::Matrix3x3(q).getColumn(0).x();  // fingers' component along pelvis +x
    };
    std::stable_sort(candidates.begin(), candidates.end(), [&](const Candidate& a, const Candidate& b) {
        return forwardness(*a.pose) > forwardness(*b.pose);
    });
    if (candidates.empty())
    {
        candidates.push_back({ "the bounding-box top grasp", std::nullopt });
    }

    // Filled for whichever attempt's pregrasp is planned to.
    ArmContext                              arm;
    MoveGroup*                              arm_group  = nullptr;
    MoveGroup*                              hand_group = nullptr;
    std::string                             arm_name;
    std::optional<geometry_msgs::msg::Pose> grasp_pose;
    geometry_msgs::msg::Pose                grasp_goal, pregrasp_goal;
    std::string                             open_shape = "open", closed_shape = "closed";
    /// A scoop's start: down at the surface in front of the edge, slid in from there.
    std::optional<geometry_msgs::msg::Pose> entry_goal;
    /// Which of scoop_clamp_approaches the other hand comes in from (-1: it cannot clamp), and
    /// how the two hands are turned from square to the edge.
    int    clamp_approach = -1;
    double scoop_turn     = 0.0;
    double clamp_turn     = 0.0;

    // Every failure below leaves the hand open and nothing attached, so a retry starts from a
    // defined state rather than part-way into a grasp. That is what makes the behavior tree's
    // retry meaningful rather than a replay.
    // Set once the hand is allowed to touch: from then on a failure leaves it on or in the
    // surface, and the hand has to come back out before contact is checked again.
    bool       contact_allowed = false;
    const auto fail = [&](const std::string& phase, const std::string& why) {
        if (hand_group)
        {
            moveToNamed(*hand_group, "open");
        }
        if (arm_group)
        {
            arm_group->detachObject(object_id);
            // Its box too: left in the world it becomes a ghost the next pick plans around.
            planning_scene_.removeCollisionObjects({ object_id });
            if (contact_allowed)
            {
                // Out the way it went in, still allowed to touch: a scoop slides back along its
                // fingers and then rises, anything else rises along its approach. Without this
                // the hand was left pressed into the table with contact checked again, every
                // later plan started "in collision", and the arm would not move at all.
                allowHandContact(arm, held_touch, true);
                const geometry_msgs::msg::Pose now =
                    arm_group->getCurrentPose(arm.grasp_frame).pose;
                geometry_msgs::msg::Pose out = now;
                if (entry_goal)
                {
                    out = slideBack(now, scoop_slide_m_, up_);
                    if (!moveLinear(*arm_group, out, arm.grasp_frame, "back out"))
                    {
                        out = now;
                    }
                }
                geometry_msgs::msg::Pose up = out;
                up.position.z += entry_goal ? scoop_hover_m_ : grasp_approach_m_;
                if (entry_goal)
                {
                    up = raised(out, up_, scoop_hover_m_);
                }
                if (!moveLinear(*arm_group, up, arm.grasp_frame, "back out"))
                {
                    RCLCPP_ERROR(
                        get_logger(),
                        "pick %s: could not take the hand back out after: %s",
                        object_id.c_str(),
                        why.c_str());
                }
            }
            allowHandContact(arm, held_touch, false);
        }
        result->success = false;
        result->message = phase + ": " + why;
        goal_handle->abort(result);
    };

    // The arms to try, in order. "auto": the hand on the side the grasp point is on first (the
    // planning frame is the pelvis, +y to the robot's left), then the other one.
    const auto arms_for = [&](double y) -> std::vector<std::string> {
        if (goal->arm != "auto")
        {
            return { goal->arm };
        }
        return y >= 0.0 ? std::vector<std::string>{ "left", "right" }
                        : std::vector<std::string>{ "right", "left" };
    };

    // A published grasp fixes WHERE the hand closes and across what (its closing axis), not
    // exactly how the hand is turned: a pinch across a wall is the same grasp with the fingers
    // tipped anywhere in the wall's plane, i.e. turned about the closing axis. Which tilt the arm
    // can reach depends on where the robot stands, so every candidate is expanded into tilts
    // (and, optionally, the hand turned over), checked with IK first -- cheap -- and only the
    // reachable ones are planned. The configured tilt comes first, then the nearest ones.
    // Where to step, for a right-hand scoop point at `grasp` (planning frame) that no placement
    // reaches: from where it is now to scoop_stance_aim, the middle of the stances the two-hand
    // scoop was measured to work from, and how far to turn to face the edge square.
    const auto stanceHint = [&](const geometry_msgs::msg::Pose& grasp) {
        tf2::Quaternion q;
        tf2::fromMsg(grasp.orientation, q);
        tf2::Vector3 heading = tf2::Matrix3x3(q).getColumn(0);
        heading              = heading - up_ * heading.dot(up_);
        const double turn    = std::atan2(heading.y(), heading.x()) * 180.0 / M_PI;
        const double ahead   = grasp.position.x - scoop_stance_aim_[0];
        const double left    = grasp.position.y - scoop_stance_aim_[1];
        char         hint[260];
        std::snprintf(
            hint,
            sizeof hint,
            "the right hand's scoop point is %.2f m ahead and %.2f m %s of the pelvis, best from "
            "about %.2f ahead and %.2f %s: step %.0f cm %s and %.0f cm %s%s",
            grasp.position.x,
            std::abs(grasp.position.y),
            grasp.position.y < 0.0 ? "right" : "left",
            scoop_stance_aim_[0],
            std::abs(scoop_stance_aim_[1]),
            scoop_stance_aim_[1] < 0.0 ? "right" : "left",
            std::abs(ahead) * 100.0,
            ahead >= 0.0 ? "forward" : "back",
            std::abs(left) * 100.0,
            left >= 0.0 ? "left" : "right",
            std::abs(turn) > 10.0 ? (turn > 0.0 ? ", and turn left to face the edge square" :
                                                  ", and turn right to face the edge square") :
                                    "");
        return std::string(hint);
    };
    struct Attempt
    {
        std::string              label;
        std::string              arm;
        std::optional<geometry_msgs::msg::Pose> grasp_pose;
        geometry_msgs::msg::Pose grasp, pregrasp;
        /// The pregrasp as joint values, from IK seeded at the current arm: planned to directly,
        /// so the arm goes to the solution nearest where it is, not whichever one OMPL samples
        /// for a pose goal (with a 7-joint arm, often a twisted elbow and wrist).
        std::map<std::string, double> pregrasp_joints;
        /// A scoop's start, below the pregrasp: see entry_goal.
        std::optional<geometry_msgs::msg::Pose> entry;
        /// A scoop the other hand can clamp: which of scoop_clamp_approaches it comes in from
        /// (-1: none, the part stays in one hand), and how both hands are turned (rad).
        int    clamp_approach{ -1 };
        double turn{ 0.0 };
        double clamp_turn{ 0.0 };
    };
    std::vector<Attempt> attempts;
    std::string          unreachable;
    std::string          scoop_stance_hint;  // where to stand instead, when no scoop is reachable
    int                  ik_checked = 0;
    for (const Candidate& c : candidates)
    {
        const double y = c.pose ? c.pose->position.y : object_pose->position.y;
        if (!c.pose)
        {
            for (const std::string& a : arms_for(y))
            {
                ArmContext ctx;
                resolveArm(a, ctx);
                Attempt at{ c.name, a, std::nullopt, graspFrameGoal(*object_pose, detection->bbox.size.z, ctx),
                            geometry_msgs::msg::Pose() };
                at.pregrasp = at.grasp;
                at.pregrasp.position.z += approach_height_m_;
                attempts.push_back(at);
            }
            continue;
        }
        int reachable_here = 0;
        const std::string cname          = graspNameOf(c.name);
        // A scoop (a hand under a panel's edge), by the hand its name says, and only from the side
        // facing the robot (fingers pointing away from it). The hand goes to the pregrasp above
        // the start, comes straight down in front of the edge to the surface the part lies on,
        // slides in along its fingers under the edge, closes the thumb down on top and lifts
        // straight up; in two hands the other hand then clamps the same edge's other half. Built
        // level with the world (levelScoop), and tried as published first, then turned about the
        // vertical (each hand, nearest first) until every pose of both hands has an IK solution:
        // where the robot stands is not exact.
        if (isScoopGrasp(cname))
        {
            // Logged, every one: a scoop silently skipped leaves the pick on whatever grasp comes
            // next, and nothing in the result says why the scoop was not it.
            if (forwardness(*c.pose) < side_grasp_min_forward_)
            {
                RCLCPP_INFO(
                    get_logger(),
                    "pick %s: %s skipped: its fingers point %.2f away from the robot (needs %.2f)",
                    object_id.c_str(),
                    c.name.c_str(),
                    forwardness(*c.pose),
                    side_grasp_min_forward_);
                continue;
            }
            // As configured (the published one first, then the nearest turns).
            const std::vector<double>& turns = scoop_turns_rad_;
            for (const std::string& a : arms_for(y))
            {
                // The clamp's offset was measured with the right hand gripping (scoop_right_near,
                // the left clamping the edge's other half, toward the panel's far end); mirrored,
                // the right hand would clamp past the panel's end. The left hand's scoop is
                // picked in one hand.
                const bool clamping =
                    scoop_two_hand_ && scoop_second_hand_ == "clamp" && a == "right";
                if (!graspIsForArm(cname, a) ||
                    static_cast<int>(attempts.size()) >= grasp_max_plans_)
                {
                    continue;
                }
                ArmContext ctx;
                resolveArm(a, ctx);
                usePinchFrame(ctx);
                ArmContext other;
                resolveArm(a == "left" ? std::string("right") : std::string("left"), other);
                usePinchFrame(other);
                const geometry_msgs::msg::Pose published =
                    levelScoop(handGoalFromGrasp(*c.pose, ctx.is_left), up_, scoop_pitch_rad_);
                tf2::Quaternion pq;
                tf2::fromMsg(published.orientation, pq);

                std::optional<Attempt> one_hand;  // the nearest variant only this hand reaches
                std::optional<Attempt> two_hands;
                int                    first_ok = 0;
                int                    tried    = 0;
                for (const double unsigned_turn : turns)
                {
                    const double turn = ctx.is_left ? -unsigned_turn : unsigned_turn;  // mirrored
                    ++tried;
                    // Turned about the vertical through the grasp point: the same point under
                    // the edge, the fingers going in at an angle.
                    geometry_msgs::msg::Pose g      = published;
                    const tf2::Quaternion    turned = tf2::Quaternion(up_, turn) * pq;
                    g.orientation                   = tf2::toMsg(turned.normalized());

                    char label[96];
                    std::snprintf(
                        label,
                        sizeof label,
                        "%s turned %+.0f deg",
                        c.name.c_str(),
                        turn * 180.0 / M_PI);
                    Attempt at{ turn == 0.0 ? c.name : std::string(label),
                                a,
                                *c.pose,
                                raised(g, up_, scoop_raise_m_),
                                geometry_msgs::msg::Pose() };
                    at.entry    = slideBack(at.grasp, scoop_slide_m_, up_);
                    at.pregrasp = raised(*at.entry, up_, scoop_hover_m_);
                    ++ik_checked;
                    const auto grasp_ik = hasIk(
                        ctx.arm_group,
                        ctx.grasp_frame,
                        at.grasp,
                        nullptr,
                        scoop_variant_ik_timeout_s_);
                    if (!grasp_ik)
                    {
                        fail(
                            Pick::Feedback::PHASE_PREGRASP,
                            "move_group's /compute_ik did not answer (MoveIt error " +
                                std::to_string(ik_broken_code_) +
                                "); is move_group running with its kinematics plugin?");
                        return;
                    }
                    const geometry_msgs::msg::Pose lifted = raised(at.grasp, up_, lift_height_m_);
                    const auto                     reach  = [&](const ArmContext&               who,
                                           const geometry_msgs::msg::Pose& p) {
                        return hasIk(
                                   who.arm_group,
                                   who.grasp_frame,
                                   p,
                                   nullptr,
                                   scoop_variant_ik_timeout_s_)
                            .value_or(false);
                    };
                    // The pregrasp's joints are the goal planned to, so they must not collide:
                    // with the hand palm up near the body's middle, KDL's first answer often
                    // swings the shoulder into the torso, and the planner then rejects the goal.
                    const char* missing = !*grasp_ik             ? "the grasp" :
                                          !reach(ctx, *at.entry) ? "the start" :
                                          !reach(ctx, lifted)    ? "the lift" :
                                          !hasIk(
                                               ctx.arm_group,
                                               ctx.grasp_frame,
                                               at.pregrasp,
                                               &at.pregrasp_joints,
                                               side_grasp_ik_timeout_s_,
                                               /*avoid_collisions=*/true)
                                                  .value_or(false) ?
                                                                "the pregrasp (collision-free)" :
                                                                nullptr;
                    if (missing)
                    {
                        RCLCPP_INFO(
                            get_logger(),
                            "pick %s:   %s with the %s hand: %s not reachable",
                            object_id.c_str(),
                            at.label.c_str(),
                            a.c_str(),
                            missing);
                        continue;
                    }
                    ++first_ok;
                    if (!one_hand)
                    {
                        one_hand = at;
                    }
                    if (!clamping)
                    {
                        break;
                    }
                    // Both lifted together, then the clamping hand's whole way in, from the first
                    // approach point it reaches.
                    if (!reach(ctx, raised(lifted, up_, scoop_support_lift_m_)))
                    {
                        continue;
                    }
                    at.turn = turn;
                    for (const double unsigned_clamp_turn : scoop_clamp_turns_rad_)
                    {
                        const double clamp_turn =
                            other.is_left ? unsigned_clamp_turn : -unsigned_clamp_turn;
                        for (std::size_t k = 0; k + 1 < scoop_clamp_pre_m_.size() && !two_hands;
                             k += 2)
                        {
                            const ClampPoses cp = clampPoses(
                                lifted,
                                turn,
                                other.is_left,
                                scoop_clamp_pre_m_[k],
                                scoop_clamp_pre_m_[k + 1],
                                clamp_turn);
                            // The clamp first: no approach point helps a clamp it cannot reach.
                            const char* clamp_missing =
                                !reach(other, cp.at)       ? "the clamp" :
                                !reach(other, cp.lifted)   ? "the lift" :
                                !reach(other, cp.in_front) ? "the start" :
                                !reach(other, cp.pre)      ? "the approach point" :
                                                             nullptr;
                            if (!clamp_missing)
                            {
                                at.clamp_approach = static_cast<int>(k / 2);
                                at.clamp_turn     = clamp_turn;
                                two_hands         = at;
                                break;
                            }
                            RCLCPP_INFO(
                                get_logger(),
                                "pick %s:   %s, the other hand clamping turned %+.0f deg from "
                                "approach "
                                "%zu: %s not reachable",
                                object_id.c_str(),
                                at.label.c_str(),
                                clamp_turn * 180.0 / M_PI,
                                k / 2,
                                clamp_missing);
                            if (clamp_missing != std::string("the start") &&
                                clamp_missing != std::string("the approach point"))
                            {
                                break;
                            }
                        }
                        if (two_hands)
                        {
                            break;
                        }
                    }
                    if (two_hands)
                    {
                        break;
                    }
                }
                const std::string other_name = a == "left" ? "right" : "left";
                const std::string clamp_note =
                    !clamping ? std::string() :
                    two_hands ? " (the " + other_name + " hand can clamp)" :
                                " (the " + other_name + " hand can clamp at none of them)";
                const std::string taking = two_hands ? "taking " + two_hands->label :
                                           one_hand ? "taking " + one_hand->label + " in one hand" :
                                                      std::string("none");
                RCLCPP_INFO(
                    get_logger(),
                    "pick %s: %s with the %s hand: %d of %d placements reachable by it%s; %s",
                    object_id.c_str(),
                    c.name.c_str(),
                    a.c_str(),
                    first_ok,
                    tried,
                    clamp_note.c_str(),
                    taking.c_str());
                if (clamping && !two_hands && cname.find("_right_") != std::string::npos)
                {
                    scoop_stance_hint = stanceHint(*c.pose);
                }
                if (two_hands || one_hand)
                {
                    attempts.push_back(two_hands ? *two_hands : *one_hand);
                    ++reachable_here;
                }
            }
            if (reachable_here == 0 && static_cast<int>(attempts.size()) < grasp_max_plans_)
            {
                char where[200];
                std::snprintf(
                    where,
                    sizeof where,
                    "%s: scoop not reachable at any placement, grasp at pelvis (%.2f, %.2f, %.2f) "
                    "m",
                    c.name.c_str(),
                    c.pose->position.x,
                    c.pose->position.y,
                    c.pose->position.z);
                unreachable += (unreachable.empty() ? "" : "; ") + std::string(where);
                if (cname.find("_right_") != std::string::npos)
                {
                    scoop_stance_hint = stanceHint(*c.pose);
                }
            }
            continue;
        }
        // A grasp from the side (a hand around a can): tried exactly as published, by the hand
        // its name says, and only with the fingers pointing away from the robot, so the right
        // hand comes in on the part's right and the left on its left. Not tilted or leaned: its
        // origin is the pinch point, beside the part's axis, and turning the hand about it would
        // move the hand off the part. A round part publishes it at several turns about its axis
        // instead. The hand comes down onto the part from above, along the world's vertical,
        // with the fingers beside it, and lifts straight up after closing.
        if (isSideGrasp(*c.pose))
        {
            if (forwardness(*c.pose) < side_grasp_min_forward_)
            {
                continue;
            }
            for (const std::string& a : arms_for(y))
            {
                if (!graspIsForArm(cname, a) ||
                    static_cast<int>(attempts.size()) >= grasp_max_plans_)
                {
                    continue;
                }
                ArmContext ctx;
                resolveArm(a, ctx);
                usePinchFrame(ctx);
                Attempt at{ c.name,
                            a,
                            *c.pose,
                            handGoalFromGrasp(*c.pose, ctx.is_left),
                            geometry_msgs::msg::Pose() };
                at.pregrasp = at.grasp;
                at.pregrasp.position.z += grasp_approach_m_;
                ++ik_checked;
                const auto grasp_ik = hasIk(
                    ctx.arm_group,
                    ctx.grasp_frame,
                    at.grasp,
                    nullptr,
                    side_grasp_ik_timeout_s_);
                if (!grasp_ik)
                {
                    fail(
                        Pick::Feedback::PHASE_PREGRASP,
                        "move_group's /compute_ik did not answer (MoveIt error " +
                            std::to_string(ik_broken_code_) +
                            "); is move_group running with its kinematics plugin?");
                    return;
                }
                if (*grasp_ik && hasIk(
                                     ctx.arm_group,
                                     ctx.grasp_frame,
                                     at.pregrasp,
                                     &at.pregrasp_joints,
                                     side_grasp_ik_timeout_s_)
                                     .value_or(false))
                {
                    attempts.push_back(at);
                    ++reachable_here;
                }
            }
            if (reachable_here == 0 && static_cast<int>(attempts.size()) < grasp_max_plans_)
            {
                char where[160];
                std::snprintf(
                    where,
                    sizeof where,
                    "%s: not reachable from above, grasp at pelvis (%.2f, %.2f, %.2f) m",
                    c.name.c_str(),
                    c.pose->position.x,
                    c.pose->position.y,
                    c.pose->position.z);
                unreachable += (unreachable.empty() ? "" : "; ") + std::string(where);
            }
            continue;
        }
        // Also leaned about the grasp's z, the axis along the edge being pinched: the hand comes
        // in over the edge at an angle instead of straight down, as a person reaching for the far
        // side of a box does. Straight down needs the wrist right above the point, which costs
        // the arm 15-20 cm of reach; leaned, the fingers reach out to it.
        std::vector<double> leans{ 0.0 };
        for (double l = grasp_tilt_step_rad_; l <= grasp_lean_max_rad_ + 1e-9 && grasp_tilt_step_rad_ > 1e-6;
             l += grasp_tilt_step_rad_)
        {
            leans.push_back(l);
            leans.push_back(-l);
        }
        std::vector<GraspVariant> variants;
        for (const double lean : leans)
        {
            geometry_msgs::msg::Pose leaned = *c.pose;
            tf2::Quaternion          q, rz;
            tf2::fromMsg(leaned.orientation, q);
            rz.setRPY(0.0, 0.0, lean);
            leaned.orientation = tf2::toMsg((q * rz).normalized());
            for (GraspVariant v :
                 graspVariants(leaned, grasp_tilt_max_rad_, grasp_tilt_step_rad_, grasp_allow_flip_, grasp_min_descent_))
            {
                v.lean = lean;
                variants.push_back(v);
            }
        }
        // Nearest to the grasp as published first: it is the one specified, the others are only
        // there for reach.
        std::stable_sort(variants.begin(), variants.end(), [](const GraspVariant& a, const GraspVariant& b) {
            return std::abs(a.tilt) + std::abs(a.lean) < std::abs(b.tilt) + std::abs(b.lean);
        });
        for (const GraspVariant& v : variants)
        {
            if (static_cast<int>(attempts.size()) >= grasp_max_plans_)
            {
                break;  // enough reachable ones to plan to
            }
            for (const std::string& a : arms_for(y))
            {
                if (!graspIsForArm(cname, a))
                {
                    continue;
                }
                ArmContext ctx;
                resolveArm(a, ctx);
                usePinchFrame(ctx);
                Attempt    at{ c.name + v.suffix(), a, v.pose, handGoalFromGrasp(v.pose, ctx.is_left),
                               geometry_msgs::msg::Pose() };
                at.pregrasp = backOff(at.grasp, grasp_approach_m_);
                // Both ends of the straight-line approach must have an IK solution (collisions
                // ignored here; the planner checks them).
                ++ik_checked;
                const auto grasp_ik = hasIk(ctx.arm_group, ctx.grasp_frame, at.grasp);
                if (!grasp_ik)
                {
                    fail(Pick::Feedback::PHASE_PREGRASP,
                         ik_broken_code_ != 0
                             ? "move_group cannot run IK for " + ctx.arm_group + " (MoveIt error " +
                                   std::to_string(ik_broken_code_) +
                                   "): its kinematics plugin did not load. kinematics.yaml uses "
                                   "pick_ik: sudo apt install ros-humble-pick-ik, then restart"
                             : "move_group's /compute_ik did not answer; is move_group running?");
                    return;
                }
                if (*grasp_ik &&
                    hasIk(ctx.arm_group, ctx.grasp_frame, at.pregrasp, &at.pregrasp_joints).value_or(false))
                {
                    attempts.push_back(at);
                    ++reachable_here;
                }
                if (static_cast<int>(attempts.size()) >= grasp_max_plans_)
                {
                    break;
                }
            }
        }
        if (reachable_here == 0 && static_cast<int>(attempts.size()) < grasp_max_plans_)
        {
            char where[160];
            std::snprintf(where, sizeof where, "%s: no tilt reachable by %s, grasp at pelvis (%.2f, %.2f, %.2f) m",
                          c.name.c_str(), goal->arm == "auto" ? "either arm" : goal->arm.c_str(),
                          c.pose->position.x, c.pose->position.y, c.pose->position.z);
            unreachable += (unreachable.empty() ? "" : "; ") + std::string(where);
        }
    }
    RCLCPP_INFO(get_logger(), "pick %s: %zu reachable grasp poses out of %d IK checks",
                object_id.c_str(), attempts.size(), ik_checked);
    if (!unreachable.empty())
    {
        RCLCPP_INFO(
            get_logger(),
            "pick %s: not reachable: %s",
            object_id.c_str(),
            unreachable.c_str());
    }

    // Two hands first: a scoop both hands reach before any held in one, whatever the candidate
    // order.
    std::stable_partition(attempts.begin(), attempts.end(), [](const Attempt& at) {
        return at.clamp_approach >= 0;
    });

    if (attempts.empty() && !scoop_stance_hint.empty())
    {
        fail(
            Pick::Feedback::PHASE_PREGRASP,
            "no scoop is reachable from here: " + scoop_stance_hint);
        return;
    }
    if (attempts.empty())
    {
        fail(Pick::Feedback::PHASE_PREGRASP,
             "out of reach at every tilt of every grasp (" + unreachable +
                 "). Stand closer: for a grasp near pelvis height the arm reaches about 0.28-0.36 m "
                 "ahead of the pelvis (x) at 0.1 m to the side, and nothing past ~0.43 m.");
        return;
    }

    // The box goes where the detection's bounding box is: for a part whose own frame is not at
    // its middle (the panel's body frame is not), the object pose would put the box off to one
    // side and leave part of the panel uncovered for the planner to swing the hand through.
    const auto box_pose = toPlanningFrame(detection->bbox.center, object_frame);
    const moveit_msgs::msg::CollisionObject object =
        publishCollisionObject(*detection, box_pose && (detection->bbox.size.x > 0.0) ? *box_pose : *object_pose);

    feedback->phase = Pick::Feedback::PHASE_PREGRASP;
    goal_handle->publish_feedback(feedback);
    bool        reached = false;
    std::string tried;
    int         planned = 0;
    for (const Attempt& at : attempts)
    {
        if (planned >= grasp_max_plans_)
        {
            break;
        }
        ++planned;
        arm_name = at.arm;
        resolveArm(arm_name, arm);
        if (at.grasp_pose)
        {
            usePinchFrame(arm);
        }
        arm_group     = groupFor(arm.arm_group);
        hand_group    = groupFor(arm.hand_group);
        grasp_pose    = at.grasp_pose;
        grasp_goal    = at.grasp;
        pregrasp_goal = at.pregrasp;
        entry_goal    = at.entry;
        clamp_approach = at.clamp_approach;
        scoop_turn     = at.turn;
        clamp_turn     = at.clamp_turn;
        RCLCPP_INFO(get_logger(), "pick %s: trying %s with the %s hand", object_id.c_str(),
                    at.label.c_str(), arm_name.c_str());

        // Opened before the approach, not after: closing on the way in would knock the object
        // off the table before the hand is around it. Reported separately from the arm move: a
        // hand that will not open is an unacquired or unpowered hand, not geometry.
        // A published grasp is a pinch: thumb across from the fingers before going down, if the
        // hand has that shape ("pinch_open" in g1.srdf). A grasp can bring its own hand states:
        // "<grasp>_open" to go down in and "<grasp>_closed" to close to (looked up by the grasp's
        // name, cut back one "_part" at a time, so "side_right_090" uses "side_open"), when the
        // hand has them -- a can is wrapped, not pinched. The bounding-box grasp opens fully.
        const auto& shapes = hand_group->getNamedTargets();
        open_shape         = "open";
        closed_shape       = "closed";
        if (grasp_pose)
        {
            const std::string name       = graspNameOf(at.label);
            const std::string own_open   = shapeFor(name, "_open", shapes);
            const std::string own_closed = shapeFor(name, "_closed", shapes);
            open_shape                   = !own_open.empty() ? own_open :
                                           std::find(shapes.begin(), shapes.end(), "pinch_open") != shapes.end() ?
                                                               "pinch_open" :
                                                               "open";
            closed_shape                 = own_closed.empty() ? "closed" : own_closed;
        }
        // Straight open for the way there, the grasp's own shape at the pregrasp: a scoop's
        // shape swings the thumb 1.2 rad across, and with the arm tucked beside the hip
        // (ready_arms.py) that puts the thumb into the hip -- every plan to it failed, "would
        // not open".
        if (!moveToNamed(*hand_group, "open"))
        {
            fail(Pick::Feedback::PHASE_PREGRASP, "the " + arm_name + " hand would not open");
            return;
        }
        const std::string what = "pregrasp (" + at.label + ")";
        if (at.pregrasp_joints.empty() ? moveTo(*arm_group, pregrasp_goal, arm.grasp_frame, what)
                                       : moveToJoints(*arm_group, at.pregrasp_joints, what))
        {
            reached = true;
            break;
        }
        // IK said reachable, so this one is blocked: the open hand or the arm on the way hits the
        // octomap or the part's box, or no path was found in time.
        RCLCPP_WARN(get_logger(), "pick %s: %s with the %s hand is reachable but the plan was blocked",
                    object_id.c_str(), at.label.c_str(), arm_name.c_str());
        tried += (tried.empty() ? "" : "; ") + at.label + " (" + arm_name + ")";
    }
    if (reached && open_shape != "open" && !moveToNamed(*hand_group, open_shape))
    {
        fail(
            Pick::Feedback::PHASE_PREGRASP,
            "the " + arm_name + " hand would not take '" + open_shape + "' at the pregrasp");
        return;
    }
    if (!reached)
    {
        fail(Pick::Feedback::PHASE_PREGRASP,
             "reachable but blocked: " + tried +
                 (unreachable.empty() ? "" : "; out of reach: " + unreachable) +
                 ". The open hand at the pregrasp hits the octomap or the part's box, or the arm "
                 "has no collision-free path there.");
        return;
    }

    feedback->phase = Pick::Feedback::PHASE_APPROACH;
    goal_handle->publish_feedback(feedback);

    // Only NOW is contact allowed, and only for the last few centimetres.
    //
    // Turning it on earlier is what let a plan route the arm straight through the table on the
    // way to the pregrasp pose: with the hand exempt from the octomap for the whole skill, a
    // path through the surface costs the planner nothing. The transit above therefore runs
    // fully collision-checked, and the exemption covers only the short descent that has to end
    // in contact.
    //
    // The object's own collision geometry is REMOVED rather than exempted. It was added so the
    // pregrasp plan would route around it; from here it is in the way, and MoveIt's documented
    // pick sequence is remove, close, attach. Attaching re-adds it as an attached body, which
    // is also what gets PointCloudOctomapUpdater's ShapeMask to stop feeding it back into the
    // octomap -- world collision objects get no such filtering, attached bodies do.
    allowHandContact(arm, { "<octomap>" }, true);
    contact_allowed = true;
    planning_scene_.removeCollisionObjects({ object_id });

    // Re-aimed at where the part is NOW, not where it was when the pick started: the tracker
    // keeps measuring it while the arm moves, and a part nudged on the way in (or simply seen
    // better from closer) moves the grasp with it. A part that moved a lot was knocked, and
    // grasping at a guess would knock it further. False when it failed the pick.
    //
    // A scoop is then re-levelled: the re-aim turns the hand with the part's estimated pose, and
    // a panel seen from closer a few degrees tilted would tip the hand flat; and the pelvis may
    // lean differently now the arm is out. The hand keeps its heading and its point, pitched
    // scoop_pitch below the world's horizontal, and starts scoop_slide_m back from it.
    const auto reaim = [&]() -> bool {
        if (grasp_pose)
        {
            if (const auto fresh = lookUpObject(object_id))
            {
                const std::string fresh_frame =
                    fresh->header.frame_id.empty() ? object_frame : fresh->header.frame_id;
                if (const auto now_pose =
                        toPlanningFrame(fresh->results.front().pose.pose, fresh_frame))
                {
                    tf2::Transform then_T, now_T, grasp_T;
                    tf2::fromMsg(aimed_object_pose, then_T);
                    tf2::fromMsg(*now_pose, now_T);
                    const tf2::Transform moved = now_T * then_T.inverse();
                    const double         shift = moved.getOrigin().length();
                    const double         turn  = moved.getRotation().getAngle();
                    if (shift > reaim_max_shift_m_)
                    {
                        fail(
                            Pick::Feedback::PHASE_APPROACH,
                            "the part moved " + std::to_string(static_cast<int>(shift * 1000)) +
                                " mm since the pick started (knocked?); not grasping at a guess");
                        return false;
                    }
                    if (shift > 0.003 || turn > 0.02)
                    {
                        RCLCPP_INFO(
                            get_logger(),
                            "pick %s: part moved %.1f mm / %.1f deg; re-aiming the grasp",
                            object_id.c_str(),
                            shift * 1000.0,
                            turn * 180.0 / M_PI);
                        tf2::fromMsg(grasp_goal, grasp_T);
                        tf2::toMsg(moved * grasp_T, grasp_goal);
                        aimed_object_pose = *now_pose;
                    }
                }
            }
        }
        if (entry_goal)
        {
            refreshUp();
            grasp_goal = levelScoop(grasp_goal, up_, scoop_pitch_rad_);
            entry_goal = slideBack(grasp_goal, scoop_slide_m_, up_);
        }
        return true;
    };
    if (!reaim())
    {
        return;
    }

    // A scoop first comes straight down in front of the edge to the surface; the approach below
    // is then the slide in under the edge, along the surface.
    if (entry_goal && !moveLinear(*arm_group, *entry_goal, arm.grasp_frame, "scoop descent"))
    {
        fail(
            Pick::Feedback::PHASE_APPROACH,
            "could not lower the hand to the surface in front of the edge");
        return;
    }
    // Down at the table, before sliding in: is the hand really pitched as planned, against the
    // world's vertical? The arm follows its joints, but a robot leaning back as it reaches tips
    // the whole hand flatter, and a flat hand's base (the white RH56E2 mount under its wrist)
    // then meets the table's edge and stops the slide -- or the fingers go in over the lip. A
    // hand too flat is re-aimed and re-levelled against the lean as it is now, once, a few
    // centimetres up and back down; still too flat, the pick stops.
    const auto pitch_now = [&]() -> std::optional<double> {
        try
        {
            refreshUp();
            const auto t = tf_buffer_->lookupTransform(
                planning_frame_,
                arm.grasp_frame,
                tf2::TimePointZero,
                tf2::durationFromSec(0.5));
            const double pitch = fingerPitch(t.transform.rotation, up_);
            RCLCPP_INFO(
                get_logger(),
                "pick %s: down at the table, the fingers point %.1f deg below the horizontal "
                "(planned %.1f)",
                object_id.c_str(),
                pitch * 180.0 / M_PI,
                scoop_pitch_rad_ * 180.0 / M_PI);
            return pitch;
        }
        catch (const tf2::TransformException& e)
        {
            RCLCPP_WARN(
                get_logger(),
                "pick %s: no TF for %s, the hand's pitch is not checked: %s",
                object_id.c_str(),
                arm.grasp_frame.c_str(),
                e.what());
            return std::nullopt;
        }
    };
    for (int round = 0; entry_goal; ++round)
    {
        const auto pitch = pitch_now();
        if (!pitch || *pitch >= scoop_pitch_rad_ - scoop_pitch_tolerance_rad_)
        {
            break;
        }
        if (round == 0)
        {
            RCLCPP_WARN(
                get_logger(),
                "pick %s: the hand is %.0f deg flatter than planned; re-levelling it for the lean",
                object_id.c_str(),
                (scoop_pitch_rad_ - *pitch) * 180.0 / M_PI);
            const geometry_msgs::msg::Pose here = arm_group->getCurrentPose(arm.grasp_frame).pose;
            if (!moveLinear(
                    *arm_group,
                    raised(here, up_, scoop_hover_m_ / 2.0),
                    arm.grasp_frame,
                    "scoop: up to re-level") ||
                !reaim() ||
                !moveLinear(
                    *arm_group,
                    raised(*entry_goal, up_, scoop_hover_m_ / 2.0),
                    arm.grasp_frame,
                    "scoop: re-levelled") ||
                !moveLinear(*arm_group, *entry_goal, arm.grasp_frame, "scoop descent, re-levelled"))
            {
                if (goal_handle->is_active())
                {
                    fail(Pick::Feedback::PHASE_APPROACH, "could not re-level the hand at the table");
                }
                return;
            }
            continue;
        }
        fail(
            Pick::Feedback::PHASE_APPROACH,
            "the hand is " +
                std::to_string(
                    static_cast<int>(std::lround((scoop_pitch_rad_ - *pitch) * 180.0 / M_PI))) +
                " deg flatter than planned at the table, re-levelled once: its base would meet the "
                "table's edge. The robot is leaning as it reaches; stand nearer the part" +
                (scoop_stance_hint.empty() ? std::string() : " (" + scoop_stance_hint + ")"));
        return;
    }

    // Straight in for a published grasp: the parts beside it are close, and a free-space plan
    // may swing the fingers through them.
    if (!(grasp_pose ? moveLinear(*arm_group, grasp_goal, arm.grasp_frame, "approach")
                     : moveTo(*arm_group, grasp_goal, arm.grasp_frame, "approach")))
    {
        fail(Pick::Feedback::PHASE_APPROACH, "could not reach the grasp pose");
        return;
    }

    // Where the hand really is versus where it was sent, both in the planning frame: the robot's
    // measured joint states through TF on one side, the goal on the other. A large error with a
    // successful plan means the arm did not follow the trajectory (or the robot model and the
    // simulator disagree); a small one with the hand beside the part means the goal itself, i.e.
    // the part's pose or the grasp, is off.
    try
    {
        const auto t = tf_buffer_->lookupTransform(planning_frame_, arm.grasp_frame, tf2::TimePointZero,
                                                   tf2::durationFromSec(0.5));
        const double ex = t.transform.translation.x - grasp_goal.position.x;
        const double ey = t.transform.translation.y - grasp_goal.position.y;
        const double ez = t.transform.translation.z - grasp_goal.position.z;
        RCLCPP_INFO(get_logger(),
                    "pick %s: %s at pelvis (%.3f, %.3f, %.3f), goal (%.3f, %.3f, %.3f): off by %.1f mm",
                    object_id.c_str(), arm.grasp_frame.c_str(), t.transform.translation.x,
                    t.transform.translation.y, t.transform.translation.z, grasp_goal.position.x,
                    grasp_goal.position.y, grasp_goal.position.z, 1000.0 * std::sqrt(ex * ex + ey * ey + ez * ez));
    }
    catch (const tf2::TransformException& e)
    {
        RCLCPP_WARN(get_logger(), "pick %s: no TF for %s: %s", object_id.c_str(), arm.grasp_frame.c_str(), e.what());
    }
    if (const auto seen = lookUpObject(object_id))
    {
        const std::string f = seen->header.frame_id.empty() ? object_frame : seen->header.frame_id;
        if (const auto p = toPlanningFrame(seen->results.front().pose.pose, f))
        {
            RCLCPP_INFO(get_logger(), "pick %s: part now at pelvis (%.3f, %.3f, %.3f), aimed at (%.3f, %.3f, %.3f)",
                        object_id.c_str(), p->position.x, p->position.y, p->position.z, aimed_object_pose.position.x,
                        aimed_object_pose.position.y, aimed_object_pose.position.z);
        }
    }

    feedback->phase = Pick::Feedback::PHASE_GRASP;
    goal_handle->publish_feedback(feedback);
    if (!moveToNamed(*hand_group, closed_shape))
    {
        fail(Pick::Feedback::PHASE_GRASP, "the hand did not close");
        return;
    }
    // Attached AFTER the hand closes, so the planner starts treating the object as part of the
    // arm at the same moment the hand does.
    //
    // Built explicitly rather than via attachObject(id, link), which promotes an object that is
    // still in the world -- and this one was removed before the approach, so there is nothing
    // left to promote. An attached body carries its own geometry, so the shape and pose come
    // from what publishCollisionObject built.
    //
    // touch_links is what tells the planner the fingers are SUPPOSED to be in contact with it.
    // Without them the attach itself reads as a collision the moment it takes effect.
    moveit_msgs::msg::AttachedCollisionObject attached;
    attached.link_name        = arm.palm_link;
    attached.object           = object;
    attached.object.operation = moveit_msgs::msg::CollisionObject::ADD;
    // The hand's own links, plus the palm and the WRIST: a cube gripped just under its top face
    // reaches past the fingers to touch it, and without the wrist in touch_links every later
    // plan starts with the attached object already in collision.
    //
    // Same set allowHandContact() exempts, and for the same reason: these are the links that
    // unavoidably touch what is being carried.
    attached.touch_links =
        hand_group->getRobotModel()->getJointModelGroup(arm.hand_group)->getLinkModelNames();
    attached.touch_links.push_back(arm.palm_link);
    const std::string touch_side = arm.is_left ? "left" : "right";
    attached.touch_links.push_back(touch_side + "_wrist_pitch_link");
    attached.touch_links.push_back(touch_side + "_wrist_yaw_link");
    attached.touch_links.push_back(touch_side + "_wrist_roll_link");
    planning_scene_.applyAttachedCollisionObject(attached);

    // Extended to the object only now that it is attached and about to be lifted OUT of the
    // surface it was resting on. Its own voxels, and the table's underneath it, are still in
    // the octomap from before the grasp -- the ShapeMask stops new clouds re-adding it, but it
    // does not erase what is already there. Without this exemption the lift starts with the
    // attached cube already in collision against the table it is still sitting on.
    allowHandContact(arm, held_touch, true);

    feedback->phase = Pick::Feedback::PHASE_LIFT;
    goal_handle->publish_feedback(feedback);
    geometry_msgs::msg::Pose lifted = grasp_goal;
    lifted.position.z += lift_height_m_;
    if (entry_goal)
    {
        lifted = raised(grasp_goal, up_, lift_height_m_);  // straight up off a level table
    }
    if (!(grasp_pose ? moveLinear(*arm_group, lifted, arm.grasp_frame, "lift")
                     : moveTo(*arm_group, lifted, arm.grasp_frame, "lift")))
    {
        fail(Pick::Feedback::PHASE_LIFT, "could not lift clear of the surface");
        return;
    }

    allowHandContact(arm, held_touch, false);

    // A scooped part hangs from one edge and sags; the other hand takes its other half too
    // (scoop_second_hand): "clamp", a second scoop in the air, fingers under the edge and the
    // thumb on top, closed; "support", flat under it, palm up. The two then lift together. If any
    // of it fails the part stays held in the one hand, as it is now: a second hand that does not
    // get there is a worse carry, dropping the part would be worse still.
    std::string    support_name;
    tf2::Transform grasp_T_support;
    std::string    support_note;
    const bool     clamp = scoop_second_hand_ == "clamp";
    if (entry_goal && scoop_two_hand_ && clamp && clamp_approach < 0)
    {
        // The candidate check found no placement both hands reach: not attempted, rather than
        // reaching for it and backing out.
        const std::string other = arm_name == "left" ? "right" : "left";
        support_note =
            "; the " + other +
            (arm_name == "left" ? std::string(" hand clamps only with the right one gripping") :
                                  std::string(" hand could not reach the other half from here")) +
            ", held in one hand" +
            (scoop_stance_hint.empty() ? std::string() : " (" + scoop_stance_hint + ")");
        RCLCPP_WARN(get_logger(), "pick %s%s", object_id.c_str(), support_note.c_str());
    }
    else if (entry_goal && scoop_two_hand_)
    {
        support_name = arm_name == "left" ? "right" : "left";
        ArmContext support;
        resolveArm(support_name, support);
        usePinchFrame(support);
        MoveGroup* support_arm  = groupFor(support.arm_group);
        MoveGroup* support_hand = groupFor(support.hand_group);

        refreshUp();  // the robot leans differently with the part held out
        if (clamp && clamp_octomap_refill_s_ > 0.0)
        {
            refreshOctomap(clamp_octomap_refill_s_);
        }
        tf2::Quaternion lq;
        tf2::fromMsg(lifted.orientation, lq);
        tf2::Vector3 heading = tf2::Matrix3x3(lq).getColumn(0);
        const tf2::Vector3 up      = up_;
        heading                    = (heading - up * heading.dot(up)).normalized();
        // A clamp from where it was checked to reach (clampPoses: on the clamping hand's own side
        // of the gripping one); a support under the part's middle.
        // Checked again now, with the part lifted and the robot leaning as it does now: the
        // combination found at the start first, then the others, the first the clamping hand
        // still reaches.
        std::optional<ClampPoses> reachable_now;
        std::string               clamp_blocked_by;
        if (clamp)
        {
            std::vector<std::pair<double, int>> combos{ { clamp_turn,
                                                          std::max(clamp_approach, 0) } };
            for (const double t : scoop_clamp_turns_rad_)
            {
                for (std::size_t k = 0; 2 * k + 1 < scoop_clamp_pre_m_.size(); ++k)
                {
                    combos.emplace_back(support.is_left ? t : -t, static_cast<int>(k));
                }
            }
            const auto reach = [&](const geometry_msgs::msg::Pose& p) {
                return hasIk(
                           support.arm_group,
                           support.grasp_frame,
                           p,
                           nullptr,
                           scoop_variant_ik_timeout_s_)
                    .value_or(false);
            };
            // The approach point is the one pose the clamping arm is PLANNED to, against the
            // octomap, the part in the other hand and the rest of the robot: reachable is not
            // enough there, it has to be reachable without touching anything -- otherwise the
            // planner spends its whole budget sampling a goal that does not exist.
            const auto reach_free = [&](const geometry_msgs::msg::Pose& p) {
                return hasIk(
                           support.arm_group,
                           support.grasp_frame,
                           p,
                           nullptr,
                           std::max(scoop_variant_ik_timeout_s_, 0.3),
                           /*avoid_collisions=*/true)
                    .value_or(false);
            };
            std::optional<ClampPoses> blocked;  // the first one only collision kept out
            for (const auto& [t, k] : combos)
            {
                const ClampPoses c = clampPoses(
                    lifted,
                    scoop_turn,
                    support.is_left,
                    scoop_clamp_pre_m_[static_cast<std::size_t>(2 * k)],
                    scoop_clamp_pre_m_[static_cast<std::size_t>(2 * k + 1)],
                    t);
                const bool reachable = reach(c.at) && reach(c.lifted) && reach(c.in_front) && reach(c.pre);
                if (reachable && !reach_free(c.pre))
                {
                    if (!blocked)
                    {
                        blocked = c;
                    }
                    continue;
                }
                if (reachable)
                {
                    RCLCPP_INFO(
                        get_logger(),
                        "pick %s: the %s hand clamps turned %+.0f deg, from approach %d",
                        object_id.c_str(),
                        support_name.c_str(),
                        t * 180.0 / M_PI,
                        k);
                    reachable_now = c;
                    break;
                }
            }
            if (!reachable_now && blocked)
            {
                // Every placement the arm reaches is blocked at its approach point. Name what
                // it hits, from the reach-only solution there, instead of letting the planner
                // time out on it.
                std::map<std::string, double> joints;
                hasIk(support.arm_group, support.grasp_frame, blocked->pre, &joints, 0.3);
                const std::vector<std::string> hits = collisionsAt(support.arm_group, joints);
                std::string list;
                for (const std::string& h : hits)
                {
                    list += (list.empty() ? "" : "; ") + h;
                }
                clamp_blocked_by = list.empty() ? std::string("collision (no contact reported)") : list;
                RCLCPP_WARN(
                    get_logger(),
                    "pick %s: every %s-hand clamp the arm reaches is blocked at its approach point; "
                    "there it hits: %s",
                    object_id.c_str(),
                    support_name.c_str(),
                    clamp_blocked_by.c_str());
            }
        }
        const int        approach = std::max(clamp_approach, 0);
        const ClampPoses cp =
            reachable_now ? *reachable_now :
                            clampPoses(
                                lifted,
                                scoop_turn,
                                support.is_left,
                                scoop_clamp_pre_m_[static_cast<std::size_t>(2 * approach)],
                                scoop_clamp_pre_m_[static_cast<std::size_t>(2 * approach + 1)],
                                clamp_turn);
        tf2::Vector3       side = up.cross(heading);
        const tf2::Vector3 to_middle(
            aimed_object_pose.position.x - grasp_goal.position.x,
            aimed_object_pose.position.y - grasp_goal.position.y,
            aimed_object_pose.position.z - grasp_goal.position.z);
        if (side.dot(to_middle) < 0.0)
        {
            side = -side;
        }
        const tf2::Vector3 under =
            tf2::Vector3(lifted.position.x, lifted.position.y, lifted.position.z) +
            heading * scoop_support_offset_[0] + side * scoop_support_offset_[1] +
            up * scoop_support_offset_[2];
        // Palm up, fingers along the heading (the frame the scoop's grasps are written in: x the
        // fingers, y out of the palm), level.
        const tf2::Vector3   across = heading.cross(up);
        const tf2::Matrix3x3 rot(
            heading.x(),
            up.x(),
            across.x(),
            heading.y(),
            up.y(),
            across.y(),
            heading.z(),
            up.z(),
            across.z());
        tf2::Quaternion sq;
        rot.getRotation(sq);
        geometry_msgs::msg::Pose hand_at;
        hand_at.position.x  = under.x();
        hand_at.position.y  = under.y();
        hand_at.position.z  = under.z();
        hand_at.orientation = tf2::toMsg(sq);
        const geometry_msgs::msg::Pose on_part =
            clamp ? cp.at : handGoalFromGrasp(hand_at, support.is_left);
        const geometry_msgs::msg::Pose both_lifted    = raised(lifted, up, scoop_support_lift_m_);
        const geometry_msgs::msg::Pose support_lifted = raised(on_part, up, scoop_support_lift_m_);

        bool held2    = false;
        bool touching = false;
        if (clamp)
        {
            // In front of the edge's other half, at its height, fingers aimed at it; in along
            // them, close, lift together. That start is low over the table, inside its voxels,
            // so the plan goes to the approach point the candidate check found reachable (back
            // and up from it), and the hand comes down from there exempt.
            const geometry_msgs::msg::Pose& in_front = cp.in_front;
            const geometry_msgs::msg::Pose& pre      = cp.pre;
            // Straight open on the way (the clamp's shape, thumb across, would hit the hip it
            // starts beside), the clamp's shape at the approach point.
            bool there =
                clamp_blocked_by.empty() && support_arm && support_hand &&
                moveToNamed(*support_hand, "open") &&
                moveTo(*support_arm, pre, support.grasp_frame, "clamp: back from the edge");
            // No fallback that plans with the hand exempt from the octomap: with the hand and
            // the wrist exempt, the planner routed the arm through the table.
            if (there && !moveToNamed(*support_hand, "scoop_open"))
            {
                RCLCPP_WARN(
                    get_logger(),
                    "pick %s: the clamping hand would not take scoop_open",
                    object_id.c_str());
                there = false;
            }
            if (there)
            {
                allowHandContact(support, held_touch, true);
                touching = true;
                held2    = moveLinear(
                            *support_arm,
                            in_front,
                            support.grasp_frame,
                            "clamp: in front of the edge") &&
                        moveLinear(
                            *support_arm,
                            on_part,
                            support.grasp_frame,
                            "clamp: in under the edge") &&
                        moveToNamed(*support_hand, "scoop_closed") &&
                        moveBothTo(arm, both_lifted, support, support_lifted, "lift in two hands");
            }
        }
        else
        {
            const geometry_msgs::msg::Pose below    = raised(on_part, up, -scoop_support_drop_m_);
            const geometry_msgs::msg::Pose in_front = slideBack(below, scoop_support_back_m_, up);
            if (support_arm && support_hand && moveToNamed(*support_hand, "open") &&
                moveTo(*support_arm, in_front, support.grasp_frame, "support: in front of the part"))
            {
                allowHandContact(support, held_touch, true);
                touching = moveLinear(
                    *support_arm,
                    below,
                    support.grasp_frame,
                    "support: in under the part");
                held2 = touching &&
                        moveLinear(
                            *support_arm,
                            on_part,
                            support.grasp_frame,
                            "support: up to the part") &&
                        moveBothTo(arm, both_lifted, support, support_lifted, "lift in two hands");
            }
        }
        if (held2)
        {
            tf2::Transform world_T_grip, world_T_support;
            tf2::fromMsg(both_lifted, world_T_grip);
            tf2::fromMsg(support_lifted, world_T_support);
            grasp_T_support = world_T_grip.inverse() * world_T_support;
            RCLCPP_INFO(
                get_logger(),
                "pick %s: held in two hands (the %s hand %s)",
                object_id.c_str(),
                support_name.c_str(),
                clamp ? "clamping the other half" : "supporting");
        }
        else
        {
            RCLCPP_WARN(
                get_logger(),
                "pick %s: the %s hand could not take the other half; held in one hand",
                object_id.c_str(),
                support_name.c_str());
            if (support_arm && touching)
            {
                const geometry_msgs::msg::Pose now =
                    support_arm->getCurrentPose(support.grasp_frame).pose;
                if (clamp)
                {
                    // Open, and back out along the fingers the way it came in.
                    if (support_hand)
                    {
                        moveToNamed(*support_hand, "scoop_open");
                    }
                    moveLinear(
                        *support_arm,
                        slideBack(now, scoop_slide_m_, up_),
                        support.grasp_frame,
                        "clamp: back out");
                }
                else
                {
                    geometry_msgs::msg::Pose down = now;
                    down.position.z -= scoop_support_drop_m_;
                    moveLinear(*support_arm, down, support.grasp_frame, "support: back down");
                    moveLinear(
                        *support_arm,
                        slideBack(down, scoop_support_back_m_, up_),
                        support.grasp_frame,
                        "support: back out");
                }
            }
            allowHandContact(support, held_touch, false);
            support_note =
                "; the " + support_name + " hand could not take the other half" +
                (clamp_blocked_by.empty() ? std::string() :
                                            " (its approach point is blocked: " + clamp_blocked_by + ")") +
                ", held in one hand";
            support_name.clear();
        }
    }

    // How it is held, so a place can put the OBJECT at a pose: grasp_frame_T_object, from the
    // grasp the hand was sent to and the object pose it was planned against.
    {
        tf2::Transform world_T_grasp, world_T_object;
        tf2::fromMsg(grasp_goal, world_T_grasp);
        tf2::fromMsg(aimed_object_pose, world_T_object);
        std::lock_guard<std::mutex> lock(held_mutex_);
        held_grasp_[object_id] = world_T_grasp.inverse() * world_T_object;
        held_arm_[object_id]   = arm_name;
        held_release_[object_id] = grasp_pose ? open_shape : std::string("open");
        held_scoop_[object_id]   = static_cast<bool>(entry_goal);
        held_yaw_free_[object_id] =
            static_cast<bool>(findHypothesis(*detection, object_id + "/symmetric/z"));
        if (!support_name.empty())
        {
            held_support_arm_[object_id]    = support_name;
            held_support_T_[object_id]      = grasp_T_support;
            held_support_clamps_[object_id] = clamp;
        }
        else
        {
            held_support_arm_.erase(object_id);
            held_support_T_.erase(object_id);
            held_support_clamps_.erase(object_id);
        }
    }
    result->success = true;
    result->message = "picked " + object_id + " with the " + arm_name + " hand" +
                      (support_name.empty() ? std::string() :
                                              (", the " + support_name + " hand " +
                                               (clamp ? "clamping the other half" : "under it"))) +
                      support_note;
    goal_handle->succeed(result);
}

void G1ManipulationServer::executePlace(const std::shared_ptr<GoalHandle<Place>>& goal_handle)
{
    const auto goal     = goal_handle->get_goal();
    auto       result   = std::make_shared<Place::Result>();
    auto       feedback = std::make_shared<Place::Feedback>();
    refreshUp();  // a scooped part slides out level with the table, not the pelvis

    // What is held, read from the scene rather than remembered, because the attachment is the
    // authority on what the hand actually has. Needed BEFORE the exemption is applied: the
    // object has to be named in it.
    std::string held_id;
    double      held_height = 0.0;
    for (const auto& [id, attached] : planning_scene_.getAttachedObjects())
    {
        if (!attached.object.primitives.empty() &&
            attached.object.primitives.front().dimensions.size() == 3)
        {
            held_id     = id;
            held_height = attached.object.primitives.front().dimensions[2];
            break;
        }
    }
    // How it is held, if this server picked it at a published grasp: then the place puts the
    // OBJECT at the target pose, and the hand goes wherever that puts it.
    std::optional<tf2::Transform> grasp_T_object;
    std::string                   picked_with;
    std::string                   release_shape = "open";
    bool                          yaw_free      = false;
    bool                          scooped       = false;
    std::string                   support_name;
    tf2::Transform                grasp_T_support;
    bool                          clamped = false;
    {
        std::lock_guard<std::mutex> lock(held_mutex_);
        if (const auto it = held_grasp_.find(held_id); it != held_grasp_.end())
        {
            grasp_T_object = it->second;
            picked_with    = held_arm_[held_id];
            release_shape  = held_release_.count(held_id) ? held_release_[held_id] : "open";
            yaw_free       = held_yaw_free_.count(held_id) && held_yaw_free_[held_id];
            scooped        = held_scoop_.count(held_id) && held_scoop_[held_id];
            if (const auto sit = held_support_arm_.find(held_id); sit != held_support_arm_.end())
            {
                support_name    = sit->second;
                grasp_T_support = held_support_T_[held_id];
                clamped = held_support_clamps_.count(held_id) && held_support_clamps_[held_id];
            }
        }
    }

    std::string arm_name = goal->arm;
    if (arm_name == "auto")
    {
        arm_name = picked_with.empty() ? "right" : picked_with;
    }
    ArmContext arm;
    if (!resolveArm(arm_name, arm))
    {
        result->success = false;
        result->message = "arm must be 'left', 'right' or 'auto', got '" + goal->arm + "'";
        goal_handle->abort(result);
        return;
    }
    if (grasp_T_object)
    {
        usePinchFrame(arm);  // it was picked at a published grasp, i.e. pinched
    }
    MoveGroup* arm_group  = groupFor(arm.arm_group);
    MoveGroup* hand_group = groupFor(arm.hand_group);

    // Held in two hands (a scoop and the other hand clamping or under it): the second hand moves
    // with the gripping one, at the pose it was picked with relative to it.
    ArmContext support;
    MoveGroup* support_arm = nullptr;
    if (!support_name.empty() && resolveArm(support_name, support))
    {
        usePinchFrame(support);
        support_arm = groupFor(support.arm_group);
    }
    const auto support_at = [&](const geometry_msgs::msg::Pose& grip) {
        tf2::Transform world_T_grip;
        tf2::fromMsg(grip, world_T_grip);
        geometry_msgs::msg::Pose out;
        tf2::toMsg(world_T_grip * grasp_T_support, out);
        return out;
    };

    // The octomap AND the object being carried: an attached body is part of the robot for
    // kinematics but is still its own collision entity, so exempting the palm it hangs off
    // does nothing for it.
    //
    // Pick already exempts both for the lift, which is the same motion in reverse.
    const std::vector<std::string> touchables =
        held_id.empty() ? std::vector<std::string>{ "<octomap>" } :
                          std::vector<std::string>{ "<octomap>", held_id };

    const auto fail = [&](const std::string& phase, const std::string& why) {
        allowHandContact(arm, touchables, false);
        if (support_arm)
        {
            allowHandContact(support, touchables, false);
        }
        result->success = false;
        result->message = phase + ": " + why;
        goal_handle->abort(result);
    };

    // The carried object may pass through the surface's voxels on the way in; the ARM may not.
    // Exempting the hand this early let plans route the arm into the bench, and the collision
    // shoved the base off its stance, putting the re-aimed descent out of reach. Same hazard
    // the pick defers its exemption for; the descent below gets the full one.
    allowHandContact(arm, touchables, true, /*include_links=*/false);

    // Where the OBJECT goes, from /objects when the goal names a detected thing.
    //
    // Prefer that over the caller's coordinate. A tree writes its drop point in the MAP frame,
    // while the base approach that parked the robot drove against /objects, published in ODOM --
    // and those two frames are only as close as AMCL is right. A realistic map->odom drift is
    // bigger than the arm's 0.04 m lateral window. Resolving the target from the stream the
    // approach used makes the two agree by construction, however far localization has drifted.
    //
    // Two kinds: "<object>/slot/<name>", a pose a detection publishes for something to sit in
    // (the object goes exactly there), or a plain object id, a surface the held object is
    // stood on top of. Returns the target in the planning frame and in the /objects frame.
    const std::string& name     = goal->surface_object_id;
    const auto         slot_at  = name.find("/slot/");
    const bool         is_slot  = slot_at != std::string::npos;
    const std::string  owner_id = is_slot ? name.substr(0, slot_at) : name;
    const auto resolve = [&]() -> std::optional<std::pair<geometry_msgs::msg::Pose, geometry_msgs::msg::Pose>> {
        const auto detection = lookUpObject(owner_id);
        if (!detection)
        {
            return std::nullopt;
        }
        const std::string frame =
            detection->header.frame_id.empty() ? objects_.header.frame_id : detection->header.frame_id;
        std::optional<geometry_msgs::msg::Pose> in_objects =
            is_slot ? findHypothesis(*detection, name) : detection->results.front().pose.pose;
        if (!in_objects)
        {
            RCLCPP_ERROR(get_logger(), "'%s' publishes no '%s'", owner_id.c_str(), name.c_str());
            return std::nullopt;
        }
        if (!is_slot)
        {
            // A detected surface reports its own centre, so the held object goes on TOP of it:
            // half the surface's height to reach its face, half the object's to stand it there.
            in_objects->position.z += 0.5 * (detection->bbox.size.z + held_height);
        }
        const auto in_planning = toPlanningFrame(*in_objects, frame);
        if (!in_planning)
        {
            return std::nullopt;
        }
        return std::make_pair(*in_planning, *in_objects);
    };

    // The grasp frame's goal for an object target, and the point the straight descent starts.
    const auto hand_goals = [&](const geometry_msgs::msg::Pose& target) {
        geometry_msgs::msg::Pose place_goal, preplace;
        if (grasp_T_object)
        {
            // world_T_grasp = world_T_object * object_T_grasp: the object exactly at the target,
            // held as it was picked. The approach is the grasp's own, in reverse for the retreat.
            tf2::Transform world_T_object;
            tf2::fromMsg(target, world_T_object);
            if (yaw_free)
            {
                // A round part may stand in its slot turned any way about its axis: pick the turn
                // that leaves the hand closest to how it is now, rather than wringing the wrist
                // round to put the part's arbitrary x where the slot's is.
                tf2::Quaternion now_q(0, 0, 0, 1);
                const auto      now = arm_group->getCurrentPose(arm.grasp_frame);
                tf2::fromMsg(now.pose.orientation, now_q);
                double         best   = 1e9;
                tf2::Transform best_T = world_T_object;
                for (int k = 0; k < 24; ++k)
                {
                    tf2::Quaternion turn;
                    turn.setRPY(0.0, 0.0, k * M_PI / 12.0);
                    const tf2::Transform turned = world_T_object * tf2::Transform(turn);
                    const tf2::Transform hand   = turned * grasp_T_object->inverse();
                    const double         angle  = hand.getRotation().angleShortestPath(now_q);
                    if (angle < best)
                    {
                        best   = angle;
                        best_T = turned;
                    }
                }
                world_T_object = best_T;
            }
            tf2::toMsg(world_T_object * grasp_T_object->inverse(), place_goal);
            // A part held from the side (or scooped from under its edge) is set down from above
            // and let go there, as it was picked: the approach runs along the world's vertical,
            // not along the fingers.
            if (scooped || isSideGrasp(place_goal))
            {
                preplace = place_goal;
                preplace.position.z += grasp_approach_m_;
            }
            else
            {
                preplace = backOff(place_goal, grasp_approach_m_);
            }
        }
        else
        {
            place_goal = graspFrameGoal(target, held_height, arm);
            preplace   = place_goal;
            preplace.position.z += approach_height_m_;
        }
        return std::make_pair(place_goal, preplace);
    };

    std::optional<geometry_msgs::msg::Pose> target;
    std::optional<geometry_msgs::msg::Point> expected;  // in the /objects frame
    if (!name.empty())
    {
        const auto resolved = resolve();
        if (!resolved)
        {
            fail(Place::Feedback::PHASE_PREPLACE, "nothing usable called '" + name + "' on /objects");
            return;
        }
        target   = resolved->first;
        expected = resolved->second.position;
    }
    else
    {
        // An empty frame means the goal is already in the planning frame. Anything else is
        // transformed rather than assumed: a target in odom treated as pelvis lands metres away,
        // and by the time that is visible the arm is already moving.
        target = toPlanningFrame(goal->pose.pose, goal->pose.header.frame_id);
    }
    if (!target)
    {
        fail(Place::Feedback::PHASE_PREPLACE, "could not transform the target pose");
        return;
    }
    auto [place_goal, preplace] = hand_goals(*target);

    feedback->phase = Place::Feedback::PHASE_PREPLACE;
    goal_handle->publish_feedback(feedback);
    // In two hands the pair goes together, straight in joint space: no plan for both_arms to a
    // pair of poses here, so keep the carry short (walk there first, then place).
    if (!(support_arm ?
              moveBothTo(arm, preplace, support, support_at(preplace), "preplace in two hands") :
              moveTo(*arm_group, preplace, arm.grasp_frame, "preplace")))
    {
        fail(Place::Feedback::PHASE_PREPLACE, "could not reach the pose above the target");
        return;
    }

    // Re-resolve before descending. Everything above was computed in the pelvis frame before the
    // arm reached out, and the reach moves the base: extending a loaded arm shifts the COM and
    // the balancing gait steps to keep up, measured at 0.165 m.
    if (!name.empty())
    {
        if (const auto fresh = resolve())
        {
            const auto [regoal, repre] = hand_goals(fresh->first);
            const double shift = std::hypot(regoal.position.x - place_goal.position.x,
                                            regoal.position.y - place_goal.position.y);
            if (shift > 0.01)
            {
                RCLCPP_INFO(get_logger(), "the base moved %.3f m during the reach; re-aiming", shift);
            }
            target     = fresh->first;
            expected   = fresh->second.position;
            place_goal = regoal;
            // The retreat returns here, so it moves with the re-aim. Left stale, the lift is
            // diagonal by however far the base walked.
            preplace = repre;
        }
    }

    feedback->phase = Place::Feedback::PHASE_LOWER;
    goal_handle->publish_feedback(feedback);
    // Clamped in two hands: the part is carried sagging -- the clamped end as far below level as
    // the clamp was placed below the gripping hand (-scoop_clamp_offset z, 5 cm) -- so kept as
    // held, the clamped end would reach the table that much before the gripped one, and a
    // set-down at the target would put the clamping hand into the table (no IK for it at all,
    // in a test). The two come down together to support_release_height_m above where the
    // clamped end touches, then the gripping hand goes the rest of the way to the target and
    // the clamping hand only to where it holds the part level: the part pivots down flat onto
    // the table, far end first. Then the clamping hand opens and slides out from under the edge
    // while the gripping one still holds the part where it was put, and goes up. The gripping
    // hand releases as for one hand.
    bool lowered = false;
    if (support_arm && clamped)
    {
        allowHandContact(arm, touchables, true);
        allowHandContact(support, touchables, true);
        const double                   sag = std::max(0.0, -scoop_clamp_offset_[2]);
        const geometry_msgs::msg::Pose near_goal =
            raised(place_goal, up_, support_release_height_m_ + sag);
        if (!moveBothTo(arm, near_goal, support, support_at(near_goal), "lower in two hands") ||
            !moveBothTo(
                arm,
                place_goal,
                support,
                raised(support_at(place_goal), up_, sag),
                "set down in two hands, level"))
        {
            fail(Place::Feedback::PHASE_LOWER, "could not lower the part in two hands");
            return;
        }
        lowered                              = true;
        MoveGroup*                     shand = groupFor(support.hand_group);
        const geometry_msgs::msg::Pose now = support_arm->getCurrentPose(support.grasp_frame).pose;
        geometry_msgs::msg::Pose       out   = slideBack(now, scoop_slide_m_, up_);
        if (!shand || !moveToNamed(*shand, "scoop_open") ||
            !moveLinear(*support_arm, out, support.grasp_frame, "clamp: slide out"))
        {
            fail(
                Place::Feedback::PHASE_RELEASE,
                "could not take the clamping hand out from under the part");
            return;
        }
        out.position.z += scoop_hover_m_;
        if (!held_id.empty())
        {
            allowHandContact(support, { held_id }, false);
        }
        if (!moveLinear(*support_arm, out, support.grasp_frame, "clamp: up"))
        {
            RCLCPP_WARN(
                get_logger(),
                "place: the clamping hand could not rise clear; left where it is");
        }
        allowHandContact(support, touchables, false);
        {
            std::lock_guard<std::mutex> lock(held_mutex_);
            held_support_arm_.erase(held_id);
            held_support_T_.erase(held_id);
            held_support_clamps_.erase(held_id);
        }
        support_arm = nullptr;
    }
    // Supported in two hands: down together to support_release_height_m above the target, then
    // the supporting hand out from under the part -- set down on it, the part would rest on that
    // hand -- and the gripping hand takes it the rest of the way alone.
    if (support_arm)
    {
        geometry_msgs::msg::Pose near_goal = place_goal;
        near_goal.position.z += support_release_height_m_;
        if (!moveBothTo(arm, near_goal, support, support_at(near_goal), "lower in two hands"))
        {
            fail(Place::Feedback::PHASE_LOWER, "could not lower the part in two hands");
            return;
        }
        allowHandContact(support, touchables, true);
        const geometry_msgs::msg::Pose now  = support_arm->getCurrentPose(support.grasp_frame).pose;
        geometry_msgs::msg::Pose       down = now;
        down.position.z -= 0.01;
        const geometry_msgs::msg::Pose out = slideBack(down, scoop_support_back_m_ + 0.05, up_);
        if (!moveLinear(*support_arm, down, support.grasp_frame, "support: off the part") ||
            !moveLinear(*support_arm, out, support.grasp_frame, "support: out from under the part"))
        {
            fail(
                Place::Feedback::PHASE_LOWER,
                "could not take the supporting hand out from under the part");
            return;
        }
        allowHandContact(support, touchables, false);
        {
            std::lock_guard<std::mutex> lock(held_mutex_);
            held_support_arm_.erase(held_id);
            held_support_T_.erase(held_id);
            held_support_clamps_.erase(held_id);
        }
        support_arm = nullptr;
    }

    // Now the hand may touch the surface: the descent ends in contact by definition.
    allowHandContact(arm, touchables, true);
    if (!lowered && !(grasp_T_object ? moveLinear(*arm_group, place_goal, arm.grasp_frame, "lower") :
                                       moveTo(*arm_group, place_goal, arm.grasp_frame, "lower")))
    {
        fail(Place::Feedback::PHASE_LOWER, "could not lower onto the target");
        return;
    }

    feedback->phase = Place::Feedback::PHASE_RELEASE;
    goal_handle->publish_feedback(feedback);
    if (!moveToNamed(*hand_group, release_shape))
    {
        fail(Place::Feedback::PHASE_RELEASE, "the hand did not open");
        return;
    }
    // Detached only after the hand is open. Detaching first would let the planner route the
    // arm through a volume the object is still occupying.
    //
    // What is held is read from the scene rather than remembered from the pick, so a place
    // still works against a server that did not do the picking -- from move_group's planning
    // scene, synchronously: MoveGroupInterface's own current state carries no attached bodies
    // (it is joint states only), and its detachObject() is a fire-and-forget topic. With that,
    // nothing was detached at all, and a part set down in two hands stayed on the gripping hand
    // and was dragged into the other hand's fingers as the hand slid out.
    std::vector<std::string> detached;
    for (const auto& [id, body] : planning_scene_.getAttachedObjects())
    {
        if (body.link_name != arm.palm_link && id != held_id)
        {
            continue;
        }
        moveit_msgs::msg::AttachedCollisionObject detach;
        detach.link_name        = body.link_name;
        detach.object.id        = id;
        detach.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
        planning_scene_.applyAttachedCollisionObject(detach);
        detached.push_back(id);
    }
    // A scooped part is set down ON the hand's fingers, and its box -- the detection's bounding
    // box, reaching back toward the robot -- is then a world object the forearm slides out
    // through. Taken out of the scene, as the pick does before its approach; the tracker sees
    // the part again where it now lies.
    if (scooped)
    {
        for (const std::string& id : detached)
        {
            moveit_msgs::msg::CollisionObject remove;
            remove.id        = id;
            remove.operation = moveit_msgs::msg::CollisionObject::REMOVE;
            planning_scene_.applyCollisionObject(remove);
        }
    }
    {
        std::lock_guard<std::mutex> lock(held_mutex_);
        held_grasp_.erase(held_id);
        held_arm_.erase(held_id);
        held_release_.erase(held_id);
        held_yaw_free_.erase(held_id);
        held_scoop_.erase(held_id);
        held_support_arm_.erase(held_id);
        held_support_T_.erase(held_id);
        held_support_clamps_.erase(held_id);
    }

    feedback->phase = Place::Feedback::PHASE_RETREAT;
    goal_handle->publish_feedback(feedback);
    // A scooped part sits on the fingers: lifting the hand would lift it again. The open hand
    // slides back out from under its edge along the surface first, still allowed to touch it,
    // and goes up from there.
    if (scooped)
    {
        const geometry_msgs::msg::Pose out = slideBack(place_goal, scoop_slide_m_, up_);
        if (!moveLinear(*arm_group, out, arm.grasp_frame, "slide out"))
        {
            fail(Place::Feedback::PHASE_RETREAT, "could not slide the hand out from under the part");
            return;
        }
        preplace = out;
        preplace.position.z += grasp_approach_m_;
    }
    // Restored BEFORE the retreat is planned, or the lift may route the open hand through what
    // was just set down. The object only: the hand is inside the surface's voxels at this height
    // by construction, so restoring `<octomap>` here would put the start state in collision.
    if (!held_id.empty())
    {
        allowHandContact(arm, { held_id }, false);
    }
    if (!(grasp_T_object ? moveLinear(*arm_group, preplace, arm.grasp_frame, "retreat")
                         : moveTo(*arm_group, preplace, arm.grasp_frame, "retreat")))
    {
        fail(Place::Feedback::PHASE_RETREAT, "could not retreat clear of the object");
        return;
    }

    allowHandContact(arm, touchables, false);

    // Did it actually land there? A successful plan says nothing about where the object ended
    // up: one release short dropped the cube on the floor with every leaf reporting success.
    //
    // Against `expected` when /objects gave the target, so both sides come from /objects and
    // the walking base cancels. A caller-supplied pose falls back to the planning frame.
    if (!held_id.empty())
    {
        if (const auto landed = lookUpObject(held_id))
        {
            const geometry_msgs::msg::Pose&  pose = landed->results.front().pose.pose;
            const geometry_msgs::msg::Point& aim  = expected ? *expected : target->position;

            std::optional<geometry_msgs::msg::Point> where = pose.position;
            if (!expected)
            {
                const std::string frame       = landed->header.frame_id.empty() ?
                                                    objects_.header.frame_id :
                                                    landed->header.frame_id;
                const auto        in_planning = toPlanningFrame(pose, frame);
                where = in_planning ? std::optional(in_planning->position) : std::nullopt;
            }
            if (where)
            {
                const double off = std::hypot(where->x - aim.x, where->y - aim.y, where->z - aim.z);
                if (off > place_tolerance_m_)
                {
                    result->success = false;
                    result->message = std::string(Place::Feedback::PHASE_RETREAT) + ": " + held_id +
                                      " ended up " + std::to_string(off) +
                                      " m from where it was placed";
                    RCLCPP_ERROR(get_logger(), "%s", result->message.c_str());
                    goal_handle->abort(result);
                    return;
                }
                RCLCPP_INFO(
                    get_logger(),
                    "%s came to rest %.3f m from the target",
                    held_id.c_str(),
                    off);
            }
        }
    }

    result->success = true;
    result->message = "placed with the " + arm_name + " hand";
    goal_handle->succeed(result);
}

void G1ManipulationServer::executeSetArmPosture(
    const std::shared_ptr<GoalHandle<SetArmPosture>>& goal_handle)
{
    const auto goal   = goal_handle->get_goal();
    auto       result = std::make_shared<SetArmPosture::Result>();

    MoveGroup* group = groupFor(goal->group);
    if (group == nullptr)
    {
        result->success = false;
        result->message = "'" + goal->group + "' is not a group this node drives";
        goal_handle->abort(result);
        return;
    }
    if (!moveToNamed(*group, goal->named_target))
    {
        result->success = false;
        result->message = "could not reach '" + goal->named_target + "'";
        goal_handle->abort(result);
        return;
    }

    result->success = true;
    result->message = goal->group + " is at " + goal->named_target;
    goal_handle->succeed(result);
}

}  // namespace g1_manipulation
