#include "g1_perception/icp_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/sample_consensus_prerejective.h>

namespace g1_perception
{

Roi projectBounds(const Eigen::AlignedBox3f& bounds, const Eigen::Isometry3f& cam_T_model, const Intrinsics& k,
                  int margin_px)
{
    float umin = 1e9f, vmin = 1e9f, umax = -1e9f, vmax = -1e9f;
    int   in_front = 0;
    for (int c = 0; c < 8; ++c)
    {
        const Eigen::Vector3f corner = cam_T_model * bounds.corner(static_cast<Eigen::AlignedBox3f::CornerType>(c));
        if (corner.z() <= 0.02f)
        {
            continue;
        }
        ++in_front;
        const float u = k.fx * corner.x() / corner.z() + k.cx;
        const float v = k.fy * corner.y() / corner.z() + k.cy;
        umin = std::min(umin, u), umax = std::max(umax, u);
        vmin = std::min(vmin, v), vmax = std::max(vmax, v);
    }
    Roi roi;
    if (in_front == 0)
    {
        return roi;
    }
    if (in_front < 8)
    {
        // Box straddles the camera plane: the projection is unbounded, use the whole image.
        roi.u1 = k.width, roi.v1 = k.height;
        return roi;
    }
    roi.u0 = std::max(0, static_cast<int>(std::floor(umin)) - margin_px);
    roi.v0 = std::max(0, static_cast<int>(std::floor(vmin)) - margin_px);
    roi.u1 = std::min(k.width, static_cast<int>(std::ceil(umax)) + margin_px);
    roi.v1 = std::min(k.height, static_cast<int>(std::ceil(vmax)) + margin_px);
    return roi;
}

void backProject(const DepthImage& depth, const Intrinsics& k, const Roi& roi, int stride, float z_min, float z_max,
                 Points& out)
{
    out.clear();
    if (roi.empty())
    {
        return;
    }
    stride = std::max(1, stride);
    const float ifx = 1.0f / k.fx, ify = 1.0f / k.fy;
    for (int v = roi.v0; v < roi.v1; v += stride)
    {
        for (int u = roi.u0; u < roi.u1; u += stride)
        {
            const float z = depth.at(u, v);
            if (!(z >= z_min && z <= z_max))  // also rejects NaN
            {
                continue;
            }
            out.emplace_back((u - k.cx) * z * ifx, (v - k.cy) * z * ify, z);
        }
    }
}

void removeExplained(const std::vector<Explained>& others, float dist, Points& pts)
{
    const float d2max = dist * dist;
    for (const auto& o : others)
    {
        if (o.model == nullptr)
        {
            continue;
        }
        const Eigen::Isometry3f   T = o.cam_T_model.inverse();
        const Eigen::AlignedBox3f box(o.model->bounds().min() - Eigen::Vector3f::Constant(dist),
                                      o.model->bounds().max() + Eigen::Vector3f::Constant(dist));
        pts.erase(std::remove_if(pts.begin(), pts.end(),
                                 [&](const Eigen::Vector3f& s) {
                                     const Eigen::Vector3f p = T * s;
                                     if (!box.contains(p))
                                     {
                                         return false;
                                     }
                                     const int idx = o.model->nearest(p);
                                     return idx >= 0 && (p - o.model->points()[idx]).squaredNorm() <= d2max;
                                 }),
                  pts.end());
    }
}

IcpResult refinePose(const ObjectModel& model, const Points& scene_cam, const Eigen::Isometry3f& init,
                     const IcpParams& params)
{
    // Solved in the MODEL frame: the scene is moved onto the model, so the model's precomputed
    // nearest-neighbour grid answers every correspondence. T maps camera -> model.
    Eigen::Isometry3f T    = init.inverse();
    const auto&       mp   = model.points();
    const auto&       mn   = model.normals();
    const bool        cull = params.backface_culling && model.normalsOriented();
    IcpResult         res;
    const int         anneal = std::max(1, params.max_iterations / 2);

    for (int it = 0; it < params.max_iterations; ++it)
    {
        const float a   = std::min(1.0f, static_cast<float>(it) / anneal);
        const float thr = params.max_corr_start + (params.max_corr_end - params.max_corr_start) * a;
        const float thr2 = thr * thr;
        const Eigen::Vector3f cam = T.translation();  // camera origin, model frame

        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
        int                         n = 0;
        for (const auto& s : scene_cam)
        {
            const Eigen::Vector3f p   = T * s;
            int                   idx = model.nearest(p);
            if (idx < 0)
            {
                continue;
            }
            if (cull && mn[idx].dot(cam - mp[idx]) < 0.0f)
            {
                // That surface faces away from the camera, so it cannot be what we see; on a thin
                // wall the face we do see is right behind it.
                idx = model.twin(idx);
                if (idx < 0 || mn[idx].dot(cam - mp[idx]) < 0.0f)
                {
                    continue;
                }
            }
            const Eigen::Vector3f& q = mp[idx];
            const Eigen::Vector3f  d = p - q;
            if (d.squaredNorm() > thr2)
            {
                continue;
            }
            const Eigen::Vector3f& nq = mn[idx];
            const float r = nq.dot(d);
            const float w = std::abs(r) <= params.huber_k ? 1.0f : params.huber_k / std::abs(r);
            Eigen::Matrix<double, 6, 1> J;
            J.head<3>() = p.cross(nq).cast<double>();
            J.tail<3>() = nq.cast<double>();
            H.noalias() += static_cast<double>(w) * J * J.transpose();
            g.noalias() += static_cast<double>(w * r) * J;
            if (params.point_weight > 0.0f)
            {
                // d(p)/d(xi) = [-[p]x, I]
                Eigen::Matrix<double, 3, 6> Jp;
                const Eigen::Vector3d       pd = p.cast<double>();
                Jp.leftCols<3>() << 0, pd.z(), -pd.y(), -pd.z(), 0, pd.x(), pd.y(), -pd.x(), 0;
                Jp.rightCols<3>().setIdentity();
                const double wp = static_cast<double>(params.point_weight * w);
                H.noalias() += wp * Jp.transpose() * Jp;
                g.noalias() += wp * Jp.transpose() * d.cast<double>();
            }
            ++n;
        }
        res.correspondences = n;
        res.iterations      = it + 1;
        if (n < 12)
        {
            break;
        }
        H.diagonal().array() += 1e-6 * H.trace() / 6.0 + 1e-12;  // keeps a degenerate view solvable
        const Eigen::Matrix<double, 6, 1> step = -H.ldlt().solve(g);
        const Eigen::Vector3f             w    = step.head<3>().cast<float>();
        const Eigen::Vector3f             v    = step.tail<3>().cast<float>();
        Eigen::Isometry3f                 delta = Eigen::Isometry3f::Identity();
        const float                       angle = w.norm();
        if (angle > 1e-12f)
        {
            delta.linear() = Eigen::AngleAxisf(angle, w / angle).toRotationMatrix();
        }
        delta.translation() = v;
        T                   = delta * T;
        if (a >= 1.0f && angle < params.converge_rot && v.norm() < params.converge_trans)
        {
            res.converged = true;
            break;
        }
    }
    res.cam_T_model = T.inverse();
    return res;
}

Score scorePose(const ObjectModel& model, const Points& scene_cam, const Eigen::Isometry3f& cam_T_model,
                float inlier_dist, float box_margin)
{
    const Eigen::Isometry3f T = cam_T_model.inverse();
    Eigen::AlignedBox3f     box(model.bounds().min() - Eigen::Vector3f::Constant(box_margin),
                                model.bounds().max() + Eigen::Vector3f::Constant(box_margin));
    const float             d2max = inlier_dist * inlier_dist;
    Score                   s;
    double                  sse = 0.0;
    for (const auto& sc : scene_cam)
    {
        const Eigen::Vector3f p = T * sc;
        if (!box.contains(p))
        {
            continue;
        }
        ++s.in_box;
        const int idx = model.nearest(p);
        if (idx < 0)
        {
            continue;
        }
        const float d2 = (p - model.points()[idx]).squaredNorm();
        if (d2 <= d2max)
        {
            ++s.inliers;
            sse += d2;
        }
    }
    s.ratio = s.in_box > 0 ? static_cast<float>(s.inliers) / s.in_box : 0.0f;
    s.rmse  = s.inliers > 0 ? static_cast<float>(std::sqrt(sse / s.inliers)) : 0.0f;
    return s;
}

void checkVisibility(const ObjectModel& model, const DepthImage& depth, const Intrinsics& k,
                     const Eigen::Isometry3f& cam_T_model, float tol, Score& score)
{
    const auto&           vp   = model.verifyPoints();
    const auto&           vn   = model.verifyNormals();
    const bool            sided = model.normalsOriented();
    const Eigen::Matrix3f R    = cam_T_model.linear();
    int                   support = 0, violate = 0, visible = 0;
    for (std::size_t i = 0; i < vp.size(); ++i)
    {
        const Eigen::Vector3f p = cam_T_model * vp[i];
        if (p.z() <= 0.05f)
        {
            continue;
        }
        if (sided && (R * vn[i]).dot(p) >= 0.0f)
        {
            continue;  // faces away from the camera
        }
        const int u = static_cast<int>(k.fx * p.x() / p.z() + k.cx + 0.5f);
        const int v = static_cast<int>(k.fy * p.y() / p.z() + k.cy + 0.5f);
        if (u < 0 || v < 0 || u >= k.width || v >= k.height)
        {
            continue;
        }
        const float z = depth.at(u, v);
        if (!(z > 0.0f) || !std::isfinite(z))
        {
            continue;  // no reading: says nothing
        }
        ++visible;
        if (std::abs(z - p.z()) <= tol)
        {
            ++support;
        }
        else if (z > p.z() + tol)
        {
            ++violate;  // the camera sees through where the model would be
        }
        // else: something in front of it -- occlusion, not evidence either way
    }
    score.visible   = visible;
    score.coverage  = visible > 0 ? static_cast<float>(support) / visible : 0.0f;
    score.violation = visible > 0 ? static_cast<float>(violate) / visible : 1.0f;
}

std::optional<Eigen::Isometry3f> globalAlign(const ObjectModel& model, const Points& scene_cam,
                                             const GlobalParams& params, unsigned seed)
{
    using PN = pcl::PointNormal;
    pcl::PointCloud<pcl::PointXYZ>::Ptr raw(new pcl::PointCloud<pcl::PointXYZ>);
    raw->reserve(scene_cam.size());
    for (const auto& p : scene_cam)
    {
        raw->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr down(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ>       vg;
    vg.setInputCloud(raw);
    vg.setLeafSize(params.voxel, params.voxel, params.voxel);
    vg.filter(*down);
    if (down->size() < 50)
    {
        return std::nullopt;
    }

    // Normals point at the camera (viewpoint = origin), matching the model's outward normals on
    // the surfaces a camera can actually see.
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>                         normals;
    ne.setNumberOfThreads(static_cast<unsigned>(std::max(1, params.threads)));
    ne.setInputCloud(down);
    ne.setRadiusSearch(params.normal_radius);
    ne.compute(normals);
    pcl::PointCloud<PN>::Ptr scene(new pcl::PointCloud<PN>);
    scene->reserve(down->size());
    for (std::size_t i = 0; i < down->size(); ++i)
    {
        const auto& n = normals[i];
        if (!std::isfinite(n.normal_x))
        {
            continue;
        }
        PN q;
        q.x = (*down)[i].x, q.y = (*down)[i].y, q.z = (*down)[i].z;
        q.normal_x = n.normal_x, q.normal_y = n.normal_y, q.normal_z = n.normal_z;
        scene->push_back(q);
    }
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_f(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::FPFHEstimationOMP<PN, PN, pcl::FPFHSignature33> fpfh;
    fpfh.setNumberOfThreads(static_cast<unsigned>(std::max(1, params.threads)));
    fpfh.setInputCloud(scene);
    fpfh.setInputNormals(scene);
    fpfh.setRadiusSearch(params.feature_radius);
    fpfh.compute(*scene_f);

    pcl::SampleConsensusPrerejective<PN, PN, pcl::FPFHSignature33> align;
    align.setInputSource(model.featureCloud());
    align.setSourceFeatures(model.features());
    align.setInputTarget(scene);
    align.setTargetFeatures(scene_f);
    align.setMaximumIterations(params.ransac_iterations);
    align.setNumberOfSamples(params.sample_size);
    align.setCorrespondenceRandomness(params.correspondence_randomness);
    align.setSimilarityThreshold(params.similarity);
    align.setMaxCorrespondenceDistance(params.max_corr);
    align.setInlierFraction(params.inlier_fraction);
    pcl::PointCloud<PN> aligned;
    std::srand(seed);  // PCL's sampler draws from rand()
    align.align(aligned);
    if (!align.hasConverged())
    {
        return std::nullopt;
    }
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.matrix()          = align.getFinalTransformation();
    return pose;
}

namespace
{

/// A private copy of a frame for the search thread, which outlives the caller's buffer.
struct DepthCopy
{
    std::vector<unsigned char> bytes;
    DepthImage                 view;

    explicit DepthCopy(const DepthImage& d) : bytes(static_cast<std::size_t>(d.step_bytes) * d.height), view(d)
    {
        std::memcpy(bytes.data(), d.data, bytes.size());
        view.data = bytes.data();
    }
};

/// How good a verified pose is, for choosing between candidates.
float quality(const Score& s) { return s.coverage - 2.0f * s.violation + 0.5f * s.ratio; }

}  // namespace

ObjectTracker::ObjectTracker(std::string name, std::shared_ptr<const ObjectModel> model, TrackerParams params)
  : name_(std::move(name)), model_(std::move(model)), params_(std::move(params))
{
}

ObjectTracker::~ObjectTracker()
{
    // The workers hold their own references to everything they read; waiting only keeps a
    // detached search from outliving the process's orderly shutdown.
    for (Search* s : { &prior_search_, &global_search_ })
    {
        if (s->valid())
        {
            s->wait();
        }
    }
}

void ObjectTracker::reset()
{
    state_      = State::kSearching;
    bad_frames_ = 0;
    searching_since_.reset();
}

bool ObjectTracker::accepted(const TrackerParams& p, const Score& s)
{
    return s.ratio >= p.min_ratio && s.inliers >= p.min_in_box && s.coverage >= p.min_coverage &&
           s.violation <= p.max_violation;
}

ObjectTracker::Output ObjectTracker::fit(const ObjectModel& model, const TrackerParams& params,
                                         const IcpParams& icp, const DepthImage& depth, const Intrinsics& k, const Eigen::Isometry3f& init,
                                         const std::vector<Explained>& others)
{
    Output    out;
    const Roi roi = projectBounds(model.bounds(), init, k, params.roi_margin_px);
    // Pick the pixel stride so the ROI yields about the point budget, then thin what is left.
    const int stride = std::max(1, static_cast<int>(std::ceil(std::sqrt(
                                       static_cast<double>(roi.area()) / std::max(1, params.max_points)))));
    Points    pts;
    backProject(depth, k, roi, stride, params.z_min, params.z_max, pts);
    removeExplained(others, params.explained_dist, pts);
    if (static_cast<int>(pts.size()) > params.max_points)
    {
        const double step = static_cast<double>(pts.size()) / params.max_points;
        Points       thin;
        thin.reserve(params.max_points);
        for (int i = 0; i < params.max_points; ++i)
        {
            thin.push_back(pts[static_cast<std::size_t>(i * step)]);
        }
        pts.swap(thin);
    }
    out.scene_points = static_cast<int>(pts.size());
    out.cam_T_model  = init;
    out.roi          = roi;
    if (pts.empty())
    {
        return out;
    }
    out.icp         = refinePose(model, pts, init, icp);
    out.cam_T_model = out.icp.cam_T_model;
    out.score       = scorePose(model, pts, out.cam_T_model, params.inlier_dist, params.box_margin);
    checkVisibility(model, depth, k, out.cam_T_model, params.free_space_tol, out.score);
    out.valid = accepted(params, out.score);
    if (params.keep_debug)
    {
        const Eigen::Isometry3f   model_T_cam = out.cam_T_model.inverse();
        const float               d2max       = params.inlier_dist * params.inlier_dist;
        const Eigen::AlignedBox3f box(model.bounds().min() - Eigen::Vector3f::Constant(params.box_margin),
                                      model.bounds().max() + Eigen::Vector3f::Constant(params.box_margin));
        out.debug_inlier.resize(pts.size());
        for (std::size_t i = 0; i < pts.size(); ++i)
        {
            // The same split the inlier ratio makes: outside the box says nothing.
            const Eigen::Vector3f p = model_T_cam * pts[i];
            if (!box.contains(p))
            {
                out.debug_inlier[i] = kDebugOutside;
                continue;
            }
            const int idx       = model.nearest(p);
            out.debug_inlier[i] = idx >= 0 && (p - model.points()[idx]).squaredNorm() <= d2max ? kDebugOnModel
                                                                                                 : kDebugOffModel;
        }
        out.debug_points = std::move(pts);
    }
    return out;
}

void ObjectTracker::startPriorSearch(const DepthImage& depth, const Intrinsics& k, const FrameContext& ctx)
{
    prior_stamp_  = ctx.stamp;
    prior_search_ = std::async(std::launch::async, [model = model_, params = params_,
                                                    frame = std::make_shared<DepthCopy>(depth), k,
                                                    priors = ctx.priors, others = ctx.others]() {
        std::optional<Eigen::Isometry3f> best;
        float                            best_q = -1e9f;
        for (const auto& prior : priors)
        {
            for (const auto& off : params.prior_offsets)
            {
                // Two passes: a wide one, since a prior may be centimetres out, then a tight one
                // from where it ended so that nearby structure cannot hold the answer off.
                Output o = fit(*model, params, params.icp, frame->view, k, prior * off, others);
                if (o.score.violation > 0.5f)
                {
                    continue;  // nothing like it here; skip the second pass
                }
                o = fit(*model, params, params.icp_track, frame->view, k, o.cam_T_model, others);
                if (o.valid && quality(o.score) > best_q)
                {
                    best_q = quality(o.score);
                    best   = o.cam_T_model;
                }
            }
        }
        return best;
    });
}

void ObjectTracker::startGlobalSearch(const DepthImage& depth, const Intrinsics& k, const FrameContext& ctx)
{
    global_stamp_       = ctx.stamp;
    const unsigned seed = 1000u * ++searches_;
    global_search_      = std::async(std::launch::async, [model = model_, params = params_,
                                                     frame = std::make_shared<DepthCopy>(depth), k,
                                                     others = ctx.others, seed]() {
        Roi whole;
        whole.u1 = k.width, whole.v1 = k.height;
        Points cloud;
        backProject(frame->view, k, whole, params.search_stride, params.z_min, params.z_max, cloud);
        removeExplained(others, params.explained_dist, cloud);
        if (params.search_region_cam)
        {
            const auto& box = *params.search_region_cam;
            cloud.erase(std::remove_if(cloud.begin(), cloud.end(),
                                       [&](const Eigen::Vector3f& p) { return !box.contains(p); }),
                        cloud.end());
        }
        std::optional<Eigen::Isometry3f> best;
        float                            best_q = -1e9f;
        for (int a = 0; a < params.global.attempts; ++a)
        {
            const auto g = globalAlign(*model, cloud, params.global, seed + a);
            if (!g)
            {
                continue;
            }
            Output o = fit(*model, params, params.icp, frame->view, k, *g, others);
            o        = fit(*model, params, params.icp_track, frame->view, k, o.cam_T_model, others);
            if (o.valid && quality(o.score) > best_q)
            {
                best_q = quality(o.score);
                best   = o.cam_T_model;
            }
        }
        return best;
    });
}

bool ObjectTracker::collect(Search& search, double stamp_then, const DepthImage& depth, const Intrinsics& k,
                            const FrameContext& ctx, Output& out)
{
    if (!search.valid() || search.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
        return false;
    }
    const auto found = search.get();
    if (!found)
    {
        return false;
    }
    // The search ran on an older frame: carry its answer through the camera's motion since.
    Eigen::Isometry3f init = *found;
    if (ctx.motion)
    {
        if (const auto m = ctx.motion(stamp_then))
        {
            init = *m * init;
        }
    }
    out = fit(*model_, params_, params_.icp_track, depth, k, init, ctx.others);
    return out.valid;
}

ObjectTracker::Output ObjectTracker::update(const DepthImage& depth, const Intrinsics& k, const FrameContext& ctx)
{
    const auto t0 = std::chrono::steady_clock::now();
    Output     out;
    if (state_ == State::kTracking)
    {
        std::vector<Eigen::Isometry3f> starts;
        starts.push_back(ctx.predicted ? *ctx.predicted : last_);
        starts.insert(starts.end(), ctx.fallbacks.begin(), ctx.fallbacks.end());
        starts.push_back(last_confirmed_);
        for (std::size_t i = 0; i < starts.size() && !out.valid; ++i)
        {
            out            = fit(*model_, params_, params_.icp_track, depth, k, starts[i], ctx.others);
            out.start_used = out.valid ? static_cast<int>(i) : -1;
        }
        if (!out.valid)
        {
            // Every start was more than the tight gate out: one wide refit before calling the
            // frame bad.
            Output wide = fit(*model_, params_, params_.icp, depth, k, starts.front(), ctx.others);
            wide        = fit(*model_, params_, params_.icp_track, depth, k, wide.cam_T_model, ctx.others);
            if (wide.valid)
            {
                out            = wide;
                out.start_used = static_cast<int>(starts.size());
            }
        }
        if (out.valid)
        {
            last_ = last_confirmed_ = out.cam_T_model;
            bad_frames_             = 0;
        }
        else
        {
            last_ = starts.front();  // keep following the prediction through a short occlusion
            if (++bad_frames_ >= params_.lost_after)
            {
                state_ = State::kSearching;
                searching_since_.reset();
            }
        }
    }
    else
    {
        if (!searching_since_)
        {
            searching_since_ = ctx.stamp;
        }
        // Two searches can be in flight: the quick one over the priors, restarted every time it
        // finishes, and the slow FPFH one, started only once the priors have had their chance.
        const bool found = collect(prior_search_, prior_stamp_, depth, k, ctx, out) ||
                           collect(global_search_, global_stamp_, depth, k, ctx, out);
        if (found)
        {
            state_      = State::kTracking;
            last_       = out.cam_T_model;
            last_confirmed_ = out.cam_T_model;
            bad_frames_ = 0;
        }
        else
        {
            if (!prior_search_.valid() && !ctx.priors.empty() &&
                ctx.stamp - last_prior_search_ >= params_.search_interval_s)
            {
                last_prior_search_ = ctx.stamp;
                startPriorSearch(depth, k, ctx);
            }
            if (!global_search_.valid() && params_.global_fallback &&
                ctx.stamp - *searching_since_ >= params_.global_after_s)
            {
                startGlobalSearch(depth, k, ctx);
            }
        }
    }
    out.state = state_;
    out.ms    = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

void ConstantVelocity::update(double stamp, const Eigen::Isometry3f& world_T)
{
    if (last_ && stamp > stamp_ && stamp - stamp_ < 0.01)
    {
        // Two stamps a few milliseconds apart turn pose noise into a large velocity: keep the
        // velocity, move the anchor.
        last_  = world_T;
        stamp_ = stamp;
        return;
    }
    if (last_ && stamp > stamp_ && stamp - stamp_ <= max_gap_)
    {
        const double          dt = stamp - stamp_;
        const Eigen::Vector3f v  = (world_T.translation() - last_->translation()) / static_cast<float>(dt);
        const Eigen::AngleAxisf aa(world_T.linear() * last_->linear().transpose());
        const Eigen::Vector3f w  = aa.axis() * aa.angle() / static_cast<float>(dt);
        const float           a  = have_velocity_ ? static_cast<float>(alpha_) : 1.0f;
        v_                       = a * v + (1.0f - a) * v_;
        w_                       = a * w + (1.0f - a) * w_;
        // Nothing on the line moves faster than this; a faster estimate is a bad pose or a
        // bad stamp, and extrapolating it would throw the next fit out of its basin.
        if (v_.norm() > 1.5f)
        {
            v_ *= 1.5f / v_.norm();
        }
        if (w_.norm() > 3.0f)
        {
            w_ *= 3.0f / w_.norm();
        }
        have_velocity_           = true;
    }
    else
    {
        v_.setZero(), w_.setZero();
        have_velocity_ = false;
    }
    last_  = world_T;
    stamp_ = stamp;
}

std::optional<Eigen::Isometry3f> ConstantVelocity::predict(double stamp) const
{
    if (!last_)
    {
        return std::nullopt;
    }
    const double dt = stamp - stamp_;
    if (!have_velocity_ || dt <= 0.0 || dt > max_gap_)
    {
        return last_;
    }
    Eigen::Isometry3f p = *last_;
    const float       a = w_.norm() * static_cast<float>(dt);
    if (a > 1e-9f)
    {
        p.linear() = Eigen::AngleAxisf(a, w_.normalized()).toRotationMatrix() * p.linear();
    }
    p.translation() += v_ * static_cast<float>(dt);
    return p;
}

}  // namespace g1_perception
