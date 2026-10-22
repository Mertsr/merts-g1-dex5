#ifndef G1_PERCEPTION__MULTI_TRACKER_HPP_
#define G1_PERCEPTION__MULTI_TRACKER_HPP_

/**
 * @file multi_tracker.hpp
 * @brief Several objects tracked in one depth stream, the way the robot needs them.
 *
 * Everything the ROS node does apart from converting messages and asking TF, so that the
 * offline benchmark drives exactly the code the robot runs:
 *
 *   - each object's pose is kept in a fixed WORLD frame (odom) and re-expressed through the
 *     camera's pose every frame, with a constant-velocity prediction on top. The head moves
 *     with every step; carried in the camera frame, that motion would look like the object
 *     jumping;
 *   - objects are fitted in the configured order, and each one's surface is removed from the
 *     depth before the others are fitted. Order matters: the pallet first, so a panel sitting
 *     in a nest with 2 mm clearance does not fit itself to the nest's walls;
 *   - priors name a frame -- a TF frame, or another tracked object -- and are resolved to the
 *     current camera frame every frame, so "nest 2 of the pallet" follows the pallet as it
 *     indexes and "in the right hand" follows the hand.
 *
 * No ROS.
 */

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "g1_perception/icp_tracker.hpp"
#include "g1_perception/pose_spec.hpp"

namespace g1_perception
{

struct ObjectConfig
{
    std::string             name;
    std::string             model_path;
    ModelOptions            model;
    TrackerParams           params;
    std::vector<FramedPose> priors;  ///< frame = a world-reachable frame or another object's name
    /// The pose reported for the object is world_T_model * model_T_output: the frame the rest
    /// of the stack calls this object's (its simulator body frame, a grasp frame), which need
    /// not be the frame its scan or CAD file happens to be authored in.
    Eigen::Isometry3f model_T_output = Eigen::Isometry3f::Identity();
    /// A reported pose outside this is rejected, and the object searched for again.
    std::optional<Region> region;
    /// Instances of one part (see expandInstances) share a group: no two of them are reported
    /// within `instance_separation` of each other, and a prior one of them already holds is not
    /// searched by the others. Empty: the object's own name.
    std::string group;
    float       instance_separation = 0.05f;
};

/// `n` copies of `c` named <name>_1 .. <name>_n, one group: several of the same part at once,
/// each free to lock onto any of the priors. The model is loaded once and shared. n <= 1 returns
/// `c` unchanged.
std::vector<ObjectConfig> expandInstances(const ObjectConfig& c, int n);

class MultiTracker
{
public:
    /// world_T_<frame> at `stamp`, or nothing when it is unknown (TF has no such transform yet).
    using FrameLookup = std::function<std::optional<Eigen::Isometry3f>(const std::string& frame, double stamp)>;

    struct Result
    {
        std::string            name;
        ObjectTracker::Output  out;
        bool                   valid = false;  ///< a pose this frame that passed verification
        Eigen::Isometry3f      world_T_output = Eigen::Isometry3f::Identity();
        Eigen::Isometry3f      cam_T_output   = Eigen::Isometry3f::Identity();
        /// The model's extent, axis-aligned in the OUTPUT frame: centre and full sizes.
        Eigen::Vector3f        box_centre = Eigen::Vector3f::Zero();
        Eigen::Vector3f        box_size   = Eigen::Vector3f::Zero();
        bool                   just_locked = false;
        bool                   just_lost   = false;
        /// The depth confirmed a fit, but it was outside the object's region or on another
        /// instance's part: dropped, and searched for again.
        bool                   rejected    = false;
        /// Debug only, while searching: every pose the prior search tries (priors x offsets),
        /// camera frame.
        std::vector<Eigen::Isometry3f> hypotheses;
    };

    /// Loads every model (seconds for a large one). Throws std::runtime_error naming the object
    /// and the reason on any failure, including duplicate names and a prior relative to the
    /// object itself. A prior relative to an object fitted LATER in the order uses that
    /// object's pose from the previous frame.
    explicit MultiTracker(std::vector<ObjectConfig> objects);

    /// One frame. `camera_frame` is passed to `lookup` for the camera's pose at earlier stamps.
    std::vector<Result> update(const DepthImage& depth, const Intrinsics& k, double stamp,
                               const Eigen::Isometry3f& world_T_cam, const std::string& camera_frame,
                               const FrameLookup& lookup);

    std::size_t size() const { return objects_.size(); }
    const ObjectConfig& config(std::size_t i) const { return objects_[i].config; }
    const ObjectModel&  model(std::size_t i) const { return objects_[i].tracker->model(); }
    void                reset();
    /// Fill the debug parts of the results (points, verdicts, hypotheses). Costs a copy of the
    /// fitted points per object per frame, so the node turns it on only while someone watches.
    void                setDebug(bool on);
    bool                debug() const { return debug_; }

private:
    struct Object
    {
        ObjectConfig                     config;
        std::unique_ptr<ObjectTracker>   tracker;
        ConstantVelocity                 motion;
        std::optional<Eigen::Isometry3f> world_T_model;  ///< last verified
        Eigen::Vector3f                  box_centre, box_size;
        bool                             was_tracking = false;
        std::string                      group;
    };
    std::vector<Object> objects_;
    bool                debug_ = false;
};

}  // namespace g1_perception

#endif  // G1_PERCEPTION__MULTI_TRACKER_HPP_
