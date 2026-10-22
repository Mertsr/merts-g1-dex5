#ifndef G1_MANIPULATION__G1_MANIPULATION_SERVER_NODE_HPP_
#define G1_MANIPULATION__G1_MANIPULATION_SERVER_NODE_HPP_

/**
 * @file g1_manipulation_server_node.hpp
 * @brief Pick, place and named-posture skills, served as actions over MoveIt.
 *
 * Adds no command path: every motion goes out through the same `move_group` the RViz panel
 * uses, onto the controllers that already own `rt/arm_sdk` and the hand topics. So the
 * one-writer rule in the control-mode rules is unaffected by this node existing.
 *
 * It also takes no control authority of its own. The arm and hands must already be acquired
 * (g1_bringup's activate_arm) before a goal will execute, and releasing them is the caller's
 * job -- for the mission that is g1_orchestration's executor, which brackets the whole run.
 * A skill that acquired authority per goal would hand it back mid-mission and drop whatever
 * the hand was holding.
 */

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <g1_msgs/action/pick.hpp>
#include <g1_msgs/action/place.hpp>
#include <g1_msgs/action/set_arm_posture.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <map>
#include <utility>
#include <memory>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <moveit_msgs/srv/get_state_validity.hpp>
#include <std_srvs/srv/empty.hpp>
#include <mutex>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <string>
#include <tf2/LinearMath/Transform.h>
#include <vector>
#include <vision_msgs/msg/detection3_d_array.hpp>

namespace g1_manipulation
{

/// The MoveIt groups and links one arm brings, resolved from the `arm` field of a goal.
struct ArmContext
{
    std::string arm_group;   ///< left_arm / right_arm
    std::string hand_group;  ///< left_hand / right_hand
    std::string palm_link;   ///< the arm group's tip, and what an object attaches to
    /// Where the hand actually closes. A frame in the URDF, so pose goals are given for it
    /// directly and nothing here does offset arithmetic.
    std::string grasp_frame;
    /// Where a thumb-to-index pinch closes ({side}_hand_pinch_frame): published grasps are
    /// planned for this one.
    std::string pinch_frame;
    /// The two hands mirror, so the palm roll flips sign with this.
    bool is_left{ false };
};

/// False if `arm` is neither "left" nor "right", leaving `out` untouched.
bool resolveArm(const std::string& arm, ArmContext& out);

/// The pose a hand's grasp frame takes for an object grasp pose. A published grasp is the RIGHT
/// hand's grasp frame at the grasp (x along the fingers, closing toward +y, thumb on +z); the left
/// hand closes toward its own -y, so it takes the same pose turned half a turn about x.
geometry_msgs::msg::Pose handGoalFromGrasp(const geometry_msgs::msg::Pose& grasp, bool is_left);

/// `pose` moved `distance` back along its own -x: for a grasp frame, out along the fingers'
/// reverse -- where the straight-line approach starts and the retreat ends.
geometry_msgs::msg::Pose backOff(const geometry_msgs::msg::Pose& pose, double distance);

/// `pose` moved `distance` straight back along its fingers' heading, the fingers' direction with
/// its component along `up` removed: where a scoop starts (in front of the edge, at the same
/// height) and where it slides back out to. `up` is the world's vertical in the pose's frame, so
/// the slide runs parallel to a level table even when that frame (the pelvis) is tilted.
geometry_msgs::msg::Pose slideBack(
    const geometry_msgs::msg::Pose& pose, double distance,
    const tf2::Vector3& up = tf2::Vector3(0.0, 0.0, 1.0));

/// `pose` moved `distance` along `up`.
geometry_msgs::msg::Pose
raised(const geometry_msgs::msg::Pose& pose, const tf2::Vector3& up, double distance);

/// A scoop hand pose rebuilt level: the fingers keep their heading about `up` and are pitched
/// exactly `pitch` (rad) below the horizontal, the palm faces up (or down, for a left hand's goal,
/// which handGoalFromGrasp turns over about the fingers -- whichever `pose` has). What a scoop
/// needs is fixed by the table, not by how the part's pose was estimated: a panel seen a few
/// degrees tilted would otherwise turn the hand flat onto the table.
geometry_msgs::msg::Pose
levelScoop(const geometry_msgs::msg::Pose& pose, const tf2::Vector3& up, double pitch);

/// How far (rad) the fingers of a grasp frame at `orientation` point below the horizontal.
double fingerPitch(const geometry_msgs::msg::Quaternion& orientation, const tf2::Vector3& up);

/// A published grasp turned about its closing axis (the grasp frame's y) by `tilt`, and turned
/// over (half a turn about the fingers) if `flipped`: the same pinch at the same point.
struct GraspVariant
{
    double                   tilt{ 0.0 };
    double                   lean{ 0.0 };  ///< set by Pick: turned about the grasp's z first
    bool                     flipped{ false };
    geometry_msgs::msg::Pose pose;
    /// " (tilt +30 deg, turned over)", or "" for the grasp as published.
    std::string suffix() const;
};

/// `grasp` and its variants, as published first, then by increasing |tilt| (+ before -), then the
/// same turned over. Variants whose fingers do not point down by at least `min_descent` (the z of
/// the approach direction, in the frame `grasp` is in) are dropped.
std::vector<GraspVariant> graspVariants(const geometry_msgs::msg::Pose& grasp, double max_tilt, double step,
                                        bool allow_flip, double min_descent);

/// The pose a detection carries under `class_id` among its results (object pose first, then the
/// grasps and slots a tracker adds), or nothing.
std::optional<geometry_msgs::msg::Pose> findHypothesis(
    const vision_msgs::msg::Detection3D& detection, const std::string& class_id);
/// The first result whose class_id starts with `prefix`, with that class_id.
std::optional<std::pair<std::string, geometry_msgs::msg::Pose>> findHypothesisPrefix(
    const vision_msgs::msg::Detection3D& detection, const std::string& prefix);

class G1ManipulationServer : public rclcpp::Node
{
public:
    explicit G1ManipulationServer(const rclcpp::NodeOptions& options);

    /**
     * @brief Builds the MoveGroupInterfaces.
     *
     * Separate from the constructor because MoveGroupInterface blocks until it has the robot
     * description and the current state, which only arrive once something is spinning this
     * node -- constructing one from inside the constructor deadlocks.
     */
    void initialize();

private:
    using Pick          = g1_msgs::action::Pick;
    using Place         = g1_msgs::action::Place;
    using SetArmPosture = g1_msgs::action::SetArmPosture;
    using MoveGroup     = moveit::planning_interface::MoveGroupInterface;

    template <typename ActionT>
    using GoalHandle = rclcpp_action::ServerGoalHandle<ActionT>;

    void executePick(const std::shared_ptr<GoalHandle<Pick>>& goal_handle);
    void executePlace(const std::shared_ptr<GoalHandle<Place>>& goal_handle);
    void executeSetArmPosture(const std::shared_ptr<GoalHandle<SetArmPosture>>& goal_handle);

    /// Latest pose for `object_id`, or nullopt if it is unknown or older than the timeout.
    std::optional<vision_msgs::msg::Detection3D> lookUpObject(const std::string& object_id);
    void onObjects(const vision_msgs::msg::Detection3DArray::ConstSharedPtr& msg);

    /// Into the planning frame, or nullopt with the reason logged. Everything a goal carries
    /// goes through here: /objects is in odom, the planner works in pelvis, and the two differ
    /// by wherever the robot is standing.
    std::optional<geometry_msgs::msg::Pose>
    toPlanningFrame(const geometry_msgs::msg::Pose& pose, const std::string& frame_id);

    /// The pose to give the arm's grasp frame so the object ends up at `object_pose`. Position
    /// passes straight through -- the grasp frame IS where the object goes -- and only the
    /// orientation is chosen here. Handed: the two hands hold at mirrored rolls.
    ///
    /// `object_height_m` is the object's FULL height: the grasp is taken just under its top
    /// face, not at its centre, or the fingers close through whatever the object is standing on.
    geometry_msgs::msg::Pose graspFrameGoal(
        const geometry_msgs::msg::Pose& object_pose, double object_height_m,
        const ArmContext& arm) const;

    /// Plans and executes so that `link` reaches `pose`. False on either failure, logged.
    bool moveTo(
        MoveGroup& group, const geometry_msgs::msg::Pose& pose, const std::string& link,
        const std::string& what);
    bool moveToNamed(MoveGroup& group, const std::string& named_target);
    /// A straight-line move of `link` to `pose` (Cartesian path), time-parameterised at the
    /// skills' velocity scaling. For the last centimetres into and out of a grasp, where a
    /// free-space plan may swing through the parts packed beside the target.
    bool moveLinear(
        MoveGroup& group, const geometry_msgs::msg::Pose& pose, const std::string& link,
        const std::string& what);

    /// Puts the object into the planning scene at its measured pose, already transformed into
    /// the planning frame, so plans route around it. Returns what it built: the world copy is
    /// removed before the grasp, and the attached body then has to carry the same geometry.
    moveit_msgs::msg::CollisionObject publishCollisionObject(
        const vision_msgs::msg::Detection3D& detection,
        const geometry_msgs::msg::Pose&      in_planning_frame);

    /**
     * @brief Lets the grasping hand touch the named things, or stops letting it.
     *
     * Grasping is contact, and the planner does not distinguish intended contact from a
     * collision. Two things are unavoidably in the way of a grasp and both have to be
     * exempted: the octomap, because the sensor has already seen the support surface and the
     * object as occupied space, and the target object's own collision geometry, which is
     * added precisely so the planner routes around it right up until the moment the hand is
     * supposed to close on it. Without this every grasp pose measures as "reachable but
     * collides".
     *
     * Scoped as narrowly as the problem allows -- the hand and the wrist that carries it, one
     * arm, one skill -- and always restored, including on every failure path, so the arm is
     * never left planning against a permanently blinded scene.
     */
    /// @param include_links  false exempts the touchables from each other only, leaving the hand
    ///        and wrist collision-checked -- what carrying an object over a surface wants.
    bool allowHandContact(
        const ArmContext& arm, const std::vector<std::string>& touchables, bool allowed,
        bool include_links = true);

    MoveGroup* groupFor(const std::string& name);

    std::map<std::string, std::shared_ptr<MoveGroup>>  groups_;
    moveit::planning_interface::PlanningSceneInterface planning_scene_;

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr objects_sub_;
    std::mutex                                                          objects_mutex_;
    vision_msgs::msg::Detection3DArray                                  objects_;

    std::unique_ptr<tf2_ros::Buffer>            tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr   get_scene_;
    rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>::SharedPtr apply_scene_;
    rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr      compute_ik_;
    rclcpp::Client<moveit_msgs::srv::GetStateValidity>::SharedPtr   check_validity_;
    int ik_broken_code_{ 0 };  ///< last /compute_ik error that was not "no solution"
    /// Whether `group` can put `link` at `pose` (planning frame), by move_group's /compute_ik:
    /// collisions ignored, or with `avoid_collisions` only a solution free of them (the arm off
    /// the torso, nothing in the scene) -- for a joint goal that is planned to directly, which
    /// the planner rejects outright when it collides. Nothing if the service did not answer.
    std::optional<bool> hasIk(
        const std::string& group, const std::string& link, const geometry_msgs::msg::Pose& pose,
        std::map<std::string, double>* joints = nullptr, double timeout_s = 0.05,
        bool avoid_collisions = false);
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr clear_octomap_;
    double octomap_refill_s_{ 1.0 };
    /// plan(), and on failure clear the octomap, let it refill and plan once more: a start
    /// state in collision with stale voxels otherwise blocks every move, including moving away.
    /// For a published (pinch) grasp: plan for the pinch frame instead of grasp_frame.
    void usePinchFrame(ArmContext& arm);
    bool planWithRecovery(MoveGroup& group, MoveGroup::Plan& plan, const std::string& what);
    bool moveToJoints(MoveGroup& group, const std::map<std::string, double>& joints, const std::string& what);
    /// Both arms at once, each to its own pose for its grasp frame: IK for each (seeded where it
    /// is), then one trajectory for both_arms, straight in joint space and timed together. For
    /// the short moves of a part held in two hands (a lift, a lower), which neither arm may make
    /// alone. Not collision-checked beyond IK: keep the moves short.
    bool moveBothTo(
        const ArmContext& a, const geometry_msgs::msg::Pose& pose_a, const ArmContext& b,
        const geometry_msgs::msg::Pose& pose_b, const std::string& what);

    rclcpp_action::Server<Pick>::SharedPtr          pick_server_;
    rclcpp_action::Server<Place>::SharedPtr         place_server_;
    rclcpp_action::Server<SetArmPosture>::SharedPtr posture_server_;

    std::string planning_frame_;
    double      object_timeout_s_{ 1.0 };
    double      grasp_height_above_top_m_{ 0.010 };
    double      place_tolerance_m_{ 0.08 };
    double      approach_height_m_{ 0.22 };
    double      lift_height_m_{ 0.15 };
    double      velocity_scaling_{ 0.3 };
    double      planning_time_s_{ 5.0 };
    // How the hand is held at the grasp. The WHERE is the grasp frame in the URDF; only the
    // orientation is a choice, and it is the one thing that depends on the surface rather than
    // on the hand.
    std::vector<double> grasp_rpy_;

    // Object grasps published with the object pose (see handGoalFromGrasp).
    double grasp_approach_m_{ 0.20 };  ///< straight-line approach and retreat length
    double reaim_max_shift_m_{ 0.04 };  ///< how far a part may move before the descent and still be followed
    double linear_min_fraction_{ 0.95 };
    double locate_timeout_s_{ 3.0 };  ///< how long a pick waits for its object to be reported
    double grasp_tilt_max_rad_{ M_PI_2 };   ///< published grasps are tried tilted up to this
    double grasp_tilt_step_rad_{ M_PI / 12 };
    bool   grasp_allow_flip_{ true };
    double grasp_lean_max_rad_{ M_PI / 3 };
    double grasp_min_descent_{ 0.25 };
    int    grasp_max_plans_{ 6 };  ///< IK-reachable grasps planned to before giving up
    /// A published grasp whose fingers are about level (a hand beside a can, not over it) is
    /// tried only if they point at least this far away from the robot (pelvis +x component of
    /// the fingers' direction): the hand then sits on the side of the part it was meant for.
    double side_grasp_min_forward_{ 0.5 };
    /// IK time per side-grasp check. The hand level beside a part near pelvis height is at the
    /// edge of the wrist's range, and KDL seeded from a hanging arm rarely finds it in the 0.05 s
    /// the other checks get.
    double side_grasp_ik_timeout_s_{ 0.5 };
    /// A scoop grasp ("scoop_*") starts this far back from the grasp along the fingers' heading,
    /// at the grasp's height, and comes down to there from scoop_hover_m_ above. A place slides
    /// the open hand back out the same distance.
    double scoop_slide_m_{ 0.11 };
    double scoop_hover_m_{ 0.08 };
    /// How much higher than published a scoop is aimed (see the parameter).
    double scoop_raise_m_{ 0.004 };
    /// How each held object is held: grasp_frame_T_object at the grasp, and by which arm.
    /// Remembered from the pick so a place can put the OBJECT where it is asked to go.
    std::mutex                              held_mutex_;
    std::map<std::string, tf2::Transform>   held_grasp_;
    std::map<std::string, std::string>      held_arm_;
    /// The hand state a held object is let go in: its grasp's "<grasp>_open" when the pick used
    /// one ("open" otherwise), so a hand wrapped around a can opens back to the shape it came
    /// down in rather than swinging its thumb through the part.
    std::map<std::string, std::string> held_release_;
    /// Objects whose detection says they are round about their own z ("<object>/symmetric/z"):
    /// a place may turn them about that axis to whatever suits the hand.
    std::map<std::string, bool> held_yaw_free_;
    /// Picked with a scoop: released by sliding the hand out from under it, not by lifting.
    std::map<std::string, bool> held_scoop_;
    /// Held in two hands: the supporting arm, and its grasp frame's goal relative to the
    /// gripping arm's (both as commanded), so a place moves the two together.
    std::map<std::string, std::string>    held_support_arm_;
    std::map<std::string, tf2::Transform> held_support_T_;
    /// ... and whether that hand clamps the part too (a second scoop) rather than only lying under it.
    std::map<std::string, bool> held_support_clamps_;
    /// A scooped part is then held by the other hand too (see the parameters): "clamp", a second
    /// scoop on the same edge in the air; "support", a flat hand under it; "none".
    bool                scoop_two_hand_{ true };
    std::string         scoop_second_hand_{ "clamp" };
    std::vector<double> scoop_clamp_offset_{ 0.002, 0.352, -0.050 };
    double              scoop_clamp_pitch_rad_{ 0.349 };
    std::vector<double> scoop_support_offset_{ -0.02, 0.314, 0.017 };
    double              scoop_support_drop_m_{ 0.08 };
    double              scoop_support_back_m_{ 0.12 };
    double              scoop_support_lift_m_{ 0.05 };
    double              support_release_height_m_{ 0.025 };

    /// Scoops are built level with the world, not the pelvis (which leans when the robot
    /// reaches): `up_` is the world's vertical in the planning frame, from TF's gravity_frame_
    /// (odom), refreshed at the start of every pick and place.
    double pick_octomap_refill_s_{ 2.5 };
    double scoop_surface_below_grasp_m_{ 0.095 };
    double scoop_surface_margin_m_{ 0.02 };
    double scoop_surface_front_m_{ 0.12 };
    double scoop_surface_thickness_m_{ 0.05 };
    bool   scoop_table_box_{ false };
    double clamp_octomap_refill_s_{ 1.5 };
    /// Clears the octomap and waits `wait_s` for the sensors to refill it.
    void refreshOctomap(double wait_s);
    /// The table under a scooped part as a collision box (see scoop_surface_*); its id, or empty.
    std::string addSurfaceSlab(const std::string& object_id, const geometry_msgs::msg::Pose& grasp);
    void        removeSurfaceSlab(const std::string& id);
    /// Removes every world collision object left over from earlier picks; returns how many.
    std::size_t clearStaleWorldObjects();
    /// What the given joint values of a group collide with, as "a / b" pairs; empty if valid
    /// or move_group did not answer.
    std::vector<std::string> collisionsAt(const std::string& group, const std::map<std::string, double>& joints);
    std::string  gravity_frame_{ "odom" };
    tf2::Vector3 up_{ 0.0, 0.0, 1.0 };
    void         refreshUp();
    /// The scoop's fingers below the horizontal, and how far the hand may really be from that
    /// once down at the table before the slide is called off (flatter, its base meets the
    /// table's edge).
    double scoop_pitch_rad_{ 0.436 };
    double scoop_pitch_tolerance_rad_{ 0.14 };
    /// Where the robot stands is not exact: a scoop is also tried with the gripping hand turned
    /// about the vertical from square to the edge (and the clamping hand, scoop_clamp_turns_rad_).
    std::vector<double> scoop_turns_rad_{ 0.0, 0.175, 0.349 };
    /// The clamping hand's approach: tried from these (back, up) points in turn.
    std::vector<double> scoop_clamp_pre_m_{ 0.12, 0.06, 0.0, 0.08 };
    /// The best stance for the two-hand scoop, as where the right hand's scoop point should be
    /// in the pelvis frame (x ahead, y left): told to the user when nothing is reachable.
    std::vector<double> scoop_stance_aim_{ 0.46, -0.16 };
    double              scoop_variant_ik_timeout_s_{ 0.1 };

    /// The clamping hand's poses for a scoop whose gripping hand is lifted to `lifted`, having
    /// come in turned by `gripping_turn` (rad, about the vertical) from square to the edge; the
    /// clamping hand comes in turned by `clamping_turn`, from the (back, up) approach point.
    struct ClampPoses
    {
        geometry_msgs::msg::Pose pre, in_front, at, lifted;
    };
    ClampPoses clampPoses(
        const geometry_msgs::msg::Pose& lifted, double gripping_turn, bool clamping_is_left,
        double pre_back, double pre_up, double clamping_turn) const;
    /// The clamping hand is also tried turned about the vertical by these (rad).
    std::vector<double> scoop_clamp_turns_rad_{ 0.0, 0.175, 0.349 };
};

}  // namespace g1_manipulation

#endif  // G1_MANIPULATION__G1_MANIPULATION_SERVER_NODE_HPP_
