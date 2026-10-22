#ifndef G1_PERCEPTION__ICP_TRACKER_HPP_
#define G1_PERCEPTION__ICP_TRACKER_HPP_

/**
 * @file icp_tracker.hpp
 * @brief Model-based 6D pose tracking from depth: global initialisation, then frame-to-model ICP.
 *
 * Per frame, per object:
 *   1. the model's bounding box at the predicted pose is projected into the image, and only that
 *      region is back-projected (the predicted pose comes from the caller -- the ROS node carries
 *      the pose in odom and re-expresses it through the head's current TF, so walking does not
 *      look like the object jumping);
 *   2. point-to-plane ICP against the model, correspondences from the model's precomputed
 *      nearest-neighbour grid (O(1) per lookup), Huber-weighted, coarse-to-fine correspondence
 *      distance, back-facing model surface excluded;
 *   3. the result is scored -- of the scene points that fall inside the model's box at that pose,
 *      how many lie on the model -- and a run of bad scores sends the object back to searching.
 *
 * Searching is FPFH + RANSAC on a downsampled frame, run on a worker thread so one object's
 * search never stalls another object's tracking. No ROS.
 */

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <utility>
#include <optional>
#include <string>
#include <vector>

#include "g1_perception/object_model.hpp"

namespace g1_perception
{

using Points = std::vector<Eigen::Vector3f>;

struct Intrinsics
{
    float fx = 0, fy = 0, cx = 0, cy = 0;
    int   width = 0, height = 0;
};

/// A depth image as it arrives: 32-bit float metres (the simulator), or 16-bit integer units
/// (a real RealSense/Orbbec, millimetres, scale 0.001). Not owned.
struct DepthImage
{
    enum class Type { kFloat32Metres, kUint16 };
    const void* data       = nullptr;
    int         width      = 0;
    int         height     = 0;
    int         step_bytes = 0;
    Type        type       = Type::kFloat32Metres;
    float       u16_scale  = 0.001f;

    float at(int u, int v) const
    {
        const auto* row = static_cast<const unsigned char*>(data) + static_cast<long>(v) * step_bytes;
        return type == Type::kFloat32Metres ? reinterpret_cast<const float*>(row)[u]
                                            : reinterpret_cast<const std::uint16_t*>(row)[u] * u16_scale;
    }
};

struct Roi
{
    int u0 = 0, v0 = 0, u1 = 0, v1 = 0;  // [u0, u1) x [v0, v1)
    bool empty() const { return u1 <= u0 || v1 <= v0; }
    long area() const { return empty() ? 0 : static_cast<long>(u1 - u0) * (v1 - v0); }
};

/// The model's box at `cam_T_model`, projected and padded by `margin_px`, clipped to the image.
/// Empty if the box is entirely behind the camera.
Roi projectBounds(const Eigen::AlignedBox3f& bounds, const Eigen::Isometry3f& cam_T_model, const Intrinsics& k,
                  int margin_px);

/// Back-projects valid pixels of `roi` (every `stride`-th) with depth in [z_min, z_max].
void backProject(const DepthImage& depth, const Intrinsics& k, const Roi& roi, int stride, float z_min, float z_max,
                 Points& out);

/// Another tracked object's surface at its current pose. Scene points it explains are removed
/// before this object is fitted: a panel seated in a fixture with 2 mm clearance otherwise sees
/// the fixture's walls as its own end faces.
struct Explained
{
    const ObjectModel* model = nullptr;
    Eigen::Isometry3f  cam_T_model = Eigen::Isometry3f::Identity();
};

/// Removes points lying within `dist` of any of `others`' surfaces.
void removeExplained(const std::vector<Explained>& others, float dist, Points& pts);

struct IcpParams
{
    int   max_iterations = 25;
    float max_corr_start = 0.03f;   ///< correspondence gate, first iteration (m)
    float max_corr_end   = 0.006f;  ///< ... annealed down to this by the middle iteration
    float huber_k        = 0.004f;  ///< residuals beyond this are down-weighted (m)
    float converge_rot   = 2e-4f;   ///< stop when a step rotates less than this (rad) ...
    float converge_trans = 5e-5f;   ///< ... and translates less than this (m)
    bool  backface_culling = true;  ///< ignored for models without oriented normals
    /// Weight of a point-to-point residual next to point-to-plane. It resists a long,
    /// constant-section part sliding along itself, but it also pulls toward the model's sample
    /// points, which biases the fit by up to a sample spacing at the part's edges (1.7 mm on the
    /// synthetic test part). Off: removing other objects' points (Explained) is what actually
    /// stopped the panel sliding, and the benchmark is as accurate and twice as fast without it.
    float point_weight = 0.0f;
};

struct IcpResult
{
    Eigen::Isometry3f cam_T_model = Eigen::Isometry3f::Identity();
    int               iterations  = 0;
    int               correspondences = 0;
    bool              converged   = false;
};

IcpResult refinePose(const ObjectModel& model, const Points& scene_cam, const Eigen::Isometry3f& init,
                     const IcpParams& params);

struct Score
{
    int   in_box  = 0;  ///< scene points inside the model's (padded) box at the pose
    int   inliers = 0;  ///< ... of which within inlier_dist of the model surface
    float ratio   = 0;  ///< inliers / in_box
    float rmse    = 0;  ///< over inliers, metres
    // Render-and-compare over the model's sparse verify points that face the camera, fall in
    // the image and have a depth reading there:
    int   visible   = 0;
    float coverage  = 0;  ///< fraction the depth confirms (within free_space_tol)
    float violation = 0;  ///< fraction where the camera sees PAST the model: it cannot be there
};

Score scorePose(const ObjectModel& model, const Points& scene_cam, const Eigen::Isometry3f& cam_T_model,
                float inlier_dist, float box_margin);

/// Fills the render-and-compare part of `score`. Occlusion (something in front of the model) is
/// neither support nor violation, so a hand over the part does not fail the check.
void checkVisibility(const ObjectModel& model, const DepthImage& depth, const Intrinsics& k,
                     const Eigen::Isometry3f& cam_T_model, float tol, Score& score);

struct GlobalParams
{
    float voxel           = 0.008f;  ///< match the model's feature_voxel
    float normal_radius   = 0.02f;
    float feature_radius  = 0.04f;
    int   ransac_iterations = 60000;
    int   sample_size     = 3;
    int   correspondence_randomness = 4;
    float similarity      = 0.85f;
    float max_corr        = 0.015f;
    float inlier_fraction = 0.2f;
    int   attempts        = 3;  ///< RANSAC runs per search; every answer is verified
    int   threads         = 2;  ///< normal and feature estimation threads
};

/// FPFH + RANSAC. Returns a coarse cam_T_model, or nothing when no consensus was found.
std::optional<Eigen::Isometry3f> globalAlign(const ObjectModel& model, const Points& scene_cam,
                                             const GlobalParams& params, unsigned seed = 0);

struct TrackerParams
{
    IcpParams    icp;        ///< searching: a hypothesis may be centimetres out
    /// Tracking: the prediction is within millimetres, so the correspondence gate starts tight.
    /// A wide gate here lets neighbouring structure (the conveyor's rails beside the pallet)
    /// pull a part off by a centimetre in a single frame.
    IcpParams    icp_track = [] {
        IcpParams p;
        p.max_corr_start = 0.012f;
        p.max_corr_end   = 0.005f;
        return p;
    }();
    GlobalParams global;
    int   max_points      = 3000;  ///< ICP input budget per frame
    int   roi_margin_px   = 20;
    float z_min           = 0.15f;
    float z_max           = 3.0f;
    float inlier_dist     = 0.006f;
    float box_margin      = 0.01f;
    float min_ratio       = 0.55f;  ///< accept / keep a pose only above this inlier ratio ...
    int   min_in_box      = 150;    ///< ... with at least this many points actually on it ...
    float min_coverage    = 0.25f;  ///< ... the depth confirming this much of its visible surface ...
    float max_violation   = 0.12f;  ///< ... and seeing past it at no more than this much
    float free_space_tol  = 0.015f;
    int   lost_after      = 5;      ///< consecutive bad frames before searching again
    int   search_stride   = 2;      ///< pixel stride for the global-search cloud
    /// Searches restart at most this often. A search over a swept prior is ~60 ICP fits, and
    /// restarting it back to back while a part is out of view would keep a core busy for nothing.
    double search_interval_s = 0.2;
    float explained_dist  = 0.004f; ///< see Explained
    /// Keep each fit's points and per-point verdicts in the Output, for visualising what the
    /// tracker sees. Off unless something is looking: it copies ~3000 points a frame.
    bool  keep_debug      = false;
    /// Every prior is also tried shifted by each of these (model frame): a pallet whose index
    /// position is unknown is found by trying the prior along its travel.
    std::vector<Eigen::Isometry3f> prior_offsets{ Eigen::Isometry3f::Identity() };
    bool  global_fallback = true;   ///< FPFH when no prior verifies ...
    double global_after_s = 1.0;    ///< ... for this long (it is slow; priors usually win first)
    /// Optional crop for FPFH searching, in the CAMERA frame of the frame being searched.
    std::optional<Eigen::AlignedBox3f> search_region_cam;
};

/// Constant-velocity prediction of a pose in a fixed (world) frame, from the poses tracking
/// returned. A pallet indexing at 0.15 m/s moves 5 mm a frame; predicting that keeps ICP inside
/// its basin on a part whose structure repeats along the direction of travel.
class ConstantVelocity
{
public:
    explicit ConstantVelocity(double smoothing = 0.5, double max_gap_s = 0.25)
      : alpha_(smoothing), max_gap_(max_gap_s)
    {
    }
    void update(double stamp, const Eigen::Isometry3f& world_T);
    /// The pose expected at `stamp`; the last pose when there is no usable velocity.
    std::optional<Eigen::Isometry3f> predict(double stamp) const;
    void reset() { last_.reset(); }

private:
    double                           alpha_, max_gap_;
    std::optional<Eigen::Isometry3f> last_;
    double                           stamp_ = 0.0;
    Eigen::Vector3f                  v_ = Eigen::Vector3f::Zero(), w_ = Eigen::Vector3f::Zero();
    bool                             have_velocity_ = false;
};

/// Everything the caller knows about a frame besides the image.
struct FrameContext
{
    double stamp = 0.0;  ///< seconds, any monotonic clock `motion` understands
    /// The caller's guess of cam_T_model for this frame (the pose carried in a world frame
    /// through the camera's motion). Without it the last pose is used.
    std::optional<Eigen::Isometry3f> predicted;
    /// How the camera moved since an earlier frame: cam_now_T_cam_then. Nothing = assume still.
    std::function<std::optional<Eigen::Isometry3f>(double stamp_then)> motion;
    /// Other objects' surfaces at their current poses (see Explained).
    std::vector<Explained> others;
    /// Where the object is likely to be, as cam_T_model for this frame: the station the pallet
    /// sits at, the pallet's nests for a panel, the hand for a carried one. Searching tries these
    /// first; only if none verifies does it fall back to FPFH over the whole frame.
    std::vector<Eigen::Isometry3f> priors;
    /// Tracking only: further starting poses tried, in order, when the fit from `predicted`
    /// fails -- the world pose without velocity, say. After them the tracker tries the last pose
    /// it confirmed IN THE CAMERA FRAME, which survives TF (odometry) that jumps or lags, and
    /// finally a wide-gate refit. Each costs one fit, only on a frame the first one failed.
    std::vector<Eigen::Isometry3f> fallbacks;
};

constexpr std::uint8_t kDebugOutside = 0, kDebugOnModel = 1, kDebugOffModel = 2;

class ObjectTracker
{
public:
    enum class State { kSearching, kTracking };

    struct Output
    {
        State             state = State::kSearching;
        bool              valid = false;  ///< a pose this frame that passed scoring
        Eigen::Isometry3f cam_T_model = Eigen::Isometry3f::Identity();
        Score             score;
        IcpResult         icp;
        int               scene_points = 0;
        double            ms = 0.0;  ///< wall time spent in update()
        /// Tracking: which start won. 0 = the prediction, 1.. = fallbacks, then the last
        /// camera-frame pose, then the wide refit; -1 = none (this frame rejected).
        int               start_used = -1;
        /// The image region the fit cut out (empty when nothing was fitted).
        Roi               roi;
        /// keep_debug only: the points ICP fitted (camera frame, other objects' points already
        /// removed), and for each a verdict: kDebugOutside the model's box, kDebugOnModel
        /// (within inlier_dist), or kDebugOffModel (in the box, off the surface: what lowers
        /// the inlier ratio).
        Points                    debug_points;
        std::vector<std::uint8_t> debug_inlier;
    };

    ObjectTracker(std::string name, std::shared_ptr<const ObjectModel> model, TrackerParams params);
    ~ObjectTracker();

    Output update(const DepthImage& depth, const Intrinsics& k, const FrameContext& ctx);

    void reset();  ///< back to searching
    const std::string& name() const { return name_; }
    State state() const { return state_; }
    const ObjectModel& model() const { return *model_; }
    TrackerParams& params() { return params_; }

    /// One ICP fit from `init` plus scoring; what tracking does every frame. Thread-safe.
    static Output fit(const ObjectModel& model, const TrackerParams& params, const IcpParams& icp,
                      const DepthImage& depth, const Intrinsics& k, const Eigen::Isometry3f& init,
                      const std::vector<Explained>& others);
    static bool accepted(const TrackerParams& params, const Score& s);

private:
    using Search = std::future<std::optional<Eigen::Isometry3f>>;
    void startPriorSearch(const DepthImage& depth, const Intrinsics& k, const FrameContext& ctx);
    void startGlobalSearch(const DepthImage& depth, const Intrinsics& k, const FrameContext& ctx);
    /// Takes a finished search's answer, carried to this frame and confirmed on it.
    bool collect(Search& search, double stamp_then, const DepthImage& depth, const Intrinsics& k,
                 const FrameContext& ctx, Output& out);

    std::string                        name_;
    std::shared_ptr<const ObjectModel> model_;
    TrackerParams                      params_;
    State                              state_ = State::kSearching;
    Eigen::Isometry3f                  last_  = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f                  last_confirmed_ = Eigen::Isometry3f::Identity();  ///< camera frame
    int                                bad_frames_ = 0;
    Search                             prior_search_, global_search_;
    double                             prior_stamp_ = 0.0, global_stamp_ = 0.0;
    double                             last_prior_search_ = -1e9;
    std::optional<double>              searching_since_;
    unsigned                           searches_ = 0;
};

}  // namespace g1_perception

#endif  // G1_PERCEPTION__ICP_TRACKER_HPP_
