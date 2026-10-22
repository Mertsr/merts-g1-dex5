// The tracking core against a synthetic scene: an asymmetric part (a plate with a block on one
// end) above a floor, rendered to an exact depth image by ray casting. No ROS, no files beyond a
// temporary STL.
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#include "g1_perception/debug_render.hpp"
#include "g1_perception/icp_tracker.hpp"
#include "g1_perception/multi_tracker.hpp"
#include "g1_perception/pose_spec.hpp"

using namespace g1_perception;

namespace
{

using Tri = std::array<Eigen::Vector3f, 3>;

// A box as 12 outward-wound triangles.
void addBox(std::vector<Tri>& tris, const Eigen::Vector3f& lo, const Eigen::Vector3f& hi)
{
    const auto c = [&](int i) {
        return Eigen::Vector3f((i & 1) ? hi.x() : lo.x(), (i & 2) ? hi.y() : lo.y(), (i & 4) ? hi.z() : lo.z());
    };
    const int faces[6][4] = { { 0, 2, 3, 1 }, { 4, 5, 7, 6 }, { 0, 1, 5, 4 }, { 2, 6, 7, 3 }, { 0, 4, 6, 2 }, { 1, 3, 7, 5 } };
    for (const auto& f : faces)
    {
        tris.push_back({ c(f[0]), c(f[1]), c(f[2]) });
        tris.push_back({ c(f[0]), c(f[2]), c(f[3]) });
    }
}

std::vector<Tri> part()
{
    std::vector<Tri> t;
    addBox(t, { -0.15f, -0.06f, 0.0f }, { 0.15f, 0.06f, 0.012f });   // plate
    addBox(t, { 0.07f, -0.06f, 0.012f }, { 0.15f, 0.0f, 0.05f });    // block on one corner
    addBox(t, { -0.15f, 0.02f, 0.012f }, { -0.11f, 0.06f, 0.03f });  // and a small one opposite
    return t;
}

std::string writeStl(const std::vector<Tri>& tris)
{
    const std::string path = ::testing::TempDir() + "g1_perception_part.stl";
    std::ofstream     out(path, std::ios::binary);
    char              header[80] = {};
    out.write(header, 80);
    const std::uint32_t n = static_cast<std::uint32_t>(tris.size());
    out.write(reinterpret_cast<const char*>(&n), 4);
    for (const auto& t : tris)
    {
        const Eigen::Vector3f nrm = (t[1] - t[0]).cross(t[2] - t[0]).normalized();
        out.write(reinterpret_cast<const char*>(nrm.data()), 12);
        for (const auto& v : t)
        {
            out.write(reinterpret_cast<const char*>(v.data()), 12);
        }
        const std::uint16_t attr = 0;
        out.write(reinterpret_cast<const char*>(&attr), 2);
    }
    return path;
}

Intrinsics camera()
{
    Intrinsics k;
    k.width = 320, k.height = 240, k.fx = k.fy = 280.0f, k.cx = 160.0f, k.cy = 120.0f;
    return k;
}

// Depth of the part at cam_T_model above a floor plane z = -0.05 (model frame), by ray casting.
std::vector<float> render(const std::vector<Tri>& tris, const Eigen::Isometry3f& cam_T_model, const Intrinsics& k)
{
    std::vector<Tri> cam;
    for (const auto& t : tris)
    {
        cam.push_back({ cam_T_model * t[0], cam_T_model * t[1], cam_T_model * t[2] });
    }
    std::vector<Tri> floor;
    addBox(floor, { -1.0f, -1.0f, -0.06f }, { 1.0f, 1.0f, -0.05f });
    for (const auto& t : floor)
    {
        cam.push_back({ cam_T_model * t[0], cam_T_model * t[1], cam_T_model * t[2] });
    }
    std::vector<float> depth(static_cast<std::size_t>(k.width) * k.height, 0.0f);
    for (int v = 0; v < k.height; ++v)
    {
        for (int u = 0; u < k.width; ++u)
        {
            const Eigen::Vector3f d((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0f);
            float                 best = 1e9f;
            for (const auto& t : cam)  // Moller-Trumbore
            {
                const Eigen::Vector3f e1 = t[1] - t[0], e2 = t[2] - t[0], p = d.cross(e2);
                const float           det = e1.dot(p);
                if (std::abs(det) < 1e-12f)
                {
                    continue;
                }
                const Eigen::Vector3f s  = -t[0];
                const float           a  = s.dot(p) / det;
                const Eigen::Vector3f q  = s.cross(e1);
                const float           b  = d.dot(q) / det;
                const float           tt = e2.dot(q) / det;
                if (a >= 0 && b >= 0 && a + b <= 1 && tt > 0 && tt < best)
                {
                    best = tt;
                }
            }
            if (best < 1e8f)
            {
                depth[static_cast<std::size_t>(v) * k.width + u] = best;  // d.z == 1, so t is z
            }
        }
    }
    return depth;
}

DepthImage view(const std::vector<float>& d, const Intrinsics& k)
{
    DepthImage img;
    img.data = d.data(), img.width = k.width, img.height = k.height, img.step_bytes = k.width * 4;
    return img;
}

// Camera 0.5 m away, looking down at the part at an angle.
Eigen::Isometry3f truePose()
{
    return poseFromXyzRpy(0.02f, -0.01f, 0.5f, 2.5f, 0.2f, 0.4f);
}

double transErrMm(const Eigen::Isometry3f& a, const Eigen::Isometry3f& b)
{
    return (a.inverse() * b).translation().norm() * 1000.0;
}

double rotErrDeg(const Eigen::Isometry3f& a, const Eigen::Isometry3f& b)
{
    return Eigen::AngleAxisf((a.inverse() * b).linear()).angle() * 180.0 / M_PI;
}

class Tracker : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        tris_  = part();
        model_ = ObjectModel::load(writeStl(tris_), ModelOptions{});
        k_     = camera();
        depth_ = render(tris_, truePose(), k_);
    }
    static std::vector<Tri>                   tris_;
    static std::shared_ptr<const ObjectModel> model_;
    static Intrinsics                         k_;
    static std::vector<float>                 depth_;
};
std::vector<Tri>                   Tracker::tris_;
std::shared_ptr<const ObjectModel> Tracker::model_;
Intrinsics                         Tracker::k_;
std::vector<float>                 Tracker::depth_;

}  // namespace

TEST(PoseSpec, ParsesPosesFramesAndSweeps)
{
    std::string err;
    const auto  p = parsePose("0.1 0.2 0.3", err);
    ASSERT_TRUE(p);
    EXPECT_TRUE(p->translation().isApprox(Eigen::Vector3f(0.1f, 0.2f, 0.3f)));
    const auto r = parsePose("0 0 0 0 0 1.5707963", err);
    ASSERT_TRUE(r);
    EXPECT_TRUE((r->linear() * Eigen::Vector3f::UnitX()).isApprox(Eigen::Vector3f::UnitY(), 1e-5f));
    const auto f = parseFramedPose("odom 1 2 3 0 0 0", err);
    ASSERT_TRUE(f);
    EXPECT_EQ(f->frame, "odom");
    const auto s = parseSweep("x 0 0.75 0.025", err);
    ASSERT_TRUE(s);
    EXPECT_EQ(s->size(), 31u);
    EXPECT_NEAR(s->back().translation().x(), 0.75f, 1e-5f);

    EXPECT_FALSE(parsePose("0.1 0.2", err));
    EXPECT_FALSE(parsePose("0.1 0.2 0.3,", err));
    EXPECT_FALSE(parseFramedPose("1 2 3", err));
    EXPECT_FALSE(parseSweep("w 0 1 0.1", err));
    EXPECT_FALSE(parseSweep("x 0 1 0", err));
}

TEST_F(Tracker, IcpConvergesFromAFewCentimetresOut)
{
    TrackerParams params;
    const auto    init = truePose() * poseFromXyzRpy(0.012f, -0.008f, 0.006f, 0.04f, -0.03f, 0.05f);
    const auto    out  = ObjectTracker::fit(*model_, params, params.icp, view(depth_, k_), k_, init, {});
    const auto    out2 = ObjectTracker::fit(*model_, params, params.icp_track, view(depth_, k_), k_, out.cam_T_model, {});
    EXPECT_LT(transErrMm(truePose(), out2.cam_T_model), 1.0);
    EXPECT_LT(rotErrDeg(truePose(), out2.cam_T_model), 0.3);
    EXPECT_TRUE(out2.valid);
    EXPECT_GT(out2.score.coverage, 0.8f);
    EXPECT_LT(out2.score.violation, 0.02f);
}

TEST_F(Tracker, VerificationRejectsAPoseWhereThePartIsNot)
{
    TrackerParams params;
    Score         s;
    // 8 cm along the plate: the camera sees floor where that pose puts the part's top.
    checkVisibility(*model_, view(depth_, k_), k_, truePose() * poseFromXyzRpy(0.0f, 0.0f, 0.08f, 0, 0, 0),
                    params.free_space_tol, s);
    EXPECT_GT(s.violation, params.max_violation);
    checkVisibility(*model_, view(depth_, k_), k_, truePose(), params.free_space_tol, s);
    EXPECT_LT(s.violation, 0.02f);
}

TEST_F(Tracker, ExplainedPointsAreRemoved)
{
    Points pts;
    Roi    whole;
    whole.u1 = k_.width, whole.v1 = k_.height;
    backProject(view(depth_, k_), k_, whole, 1, 0.1f, 3.0f, pts);
    const std::size_t before = pts.size();
    removeExplained({ { model_.get(), truePose() } }, 0.004f, pts);
    EXPECT_LT(pts.size(), before);
    EXPECT_GT(pts.size(), before / 4);  // the floor stays
    for (const auto& p : pts)
    {
        const Eigen::Vector3f q = truePose().inverse() * p;
        const int             i = model_->nearest(q);
        if (i >= 0)
        {
            EXPECT_GT((q - model_->points()[i]).norm(), 0.0035f);
        }
    }
}

TEST_F(Tracker, LocksFromAPriorAndKeepsTracking)
{
    TrackerParams params;
    params.global_fallback = false;
    ObjectTracker tracker("part", model_, params);
    FrameContext  ctx;
    // A prior 2 cm and 5 degrees off, as a station pose in a map would be.
    ctx.priors = { truePose() * poseFromXyzRpy(-0.015f, 0.01f, 0.0f, 0.0f, 0.0f, 0.08f) };
    const auto d = view(depth_, k_);
    bool       locked = false;
    for (int f = 0; f < 100 && !locked; ++f)
    {
        ctx.stamp = f / 30.0;
        locked    = tracker.update(d, k_, ctx).valid;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(locked);
    ctx.priors.clear();
    const auto out = tracker.update(d, k_, ctx);
    ASSERT_TRUE(out.valid);
    EXPECT_LT(transErrMm(truePose(), out.cam_T_model), 1.0);
    EXPECT_EQ(tracker.state(), ObjectTracker::State::kTracking);
}

TEST_F(Tracker, AWrongPriorNeverLocks)
{
    TrackerParams params;
    params.global_fallback = false;
    ObjectTracker tracker("part", model_, params);
    FrameContext  ctx;
    ctx.priors   = { truePose() * poseFromXyzRpy(0.25f, 0.0f, 0.0f, 0, 0, 0) };  // beside the part
    const auto d = view(depth_, k_);
    for (int f = 0; f < 30; ++f)
    {
        ctx.stamp = f / 30.0;
        EXPECT_FALSE(tracker.update(d, k_, ctx).valid);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

TEST(ConstantVelocity, PredictsSteadyMotion)
{
    ConstantVelocity cv;
    for (int i = 0; i < 5; ++i)
    {
        cv.update(i / 30.0, poseFromXyzRpy(0.005f * i, 0, 0, 0, 0, 0.01f * i));
    }
    const auto p = cv.predict(5 / 30.0);
    ASSERT_TRUE(p);
    EXPECT_NEAR(p->translation().x(), 0.025f, 1e-4f);
    EXPECT_NEAR(Eigen::AngleAxisf(p->linear()).angle(), 0.05f, 1e-4f);
    // A gap longer than max_gap_s: no velocity to trust, the last pose is the guess.
    const auto late = cv.predict(5.0);
    ASSERT_TRUE(late);
    EXPECT_NEAR(late->translation().x(), 0.02f, 1e-5f);
}

TEST_F(Tracker, MultiTrackerResolvesPriorsAndReportsTheOutputFrame)
{
    ObjectConfig c;
    c.name       = "part";
    c.model_path = ::testing::TempDir() + "g1_perception_part.stl";
    c.params.global_fallback = false;
    // World = camera here; the prior hangs off a "station" frame the lookup knows. Offset toward
    // -x: from +x this view converges 9 mm short, because the part's -x faces point away from
    // the camera and nothing visible resists sliding that way. That is a property of the view,
    // which no ICP variant tried (point-to-point terms, wider gates, more iterations) changes.
    const Eigen::Isometry3f world_T_station = truePose() * poseFromXyzRpy(-0.015f, 0.01f, 0.0f, 0, 0, 0.08f);
    c.priors.push_back({ "station", Eigen::Isometry3f::Identity() });
    c.model_T_output = poseFromXyzRpy(0.0f, 0.0f, 0.025f, 0, 0, 0);
    MultiTracker mt({ c });
    const auto   lookup = [&](const std::string& frame, double) -> std::optional<Eigen::Isometry3f> {
        if (frame == "station")
        {
            return world_T_station;
        }
        if (frame == "camera")
        {
            return Eigen::Isometry3f::Identity();
        }
        return std::nullopt;
    };
    const auto                  d = view(depth_, k_);
    std::vector<MultiTracker::Result> r;
    for (int f = 0; f < 100; ++f)
    {
        r = mt.update(d, k_, f / 30.0, Eigen::Isometry3f::Identity(), "camera", lookup);
        if (r[0].valid)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(r[0].valid);
    EXPECT_TRUE(r[0].just_locked);
    EXPECT_LT(transErrMm(truePose() * c.model_T_output, r[0].world_T_output), 1.0);
    EXPECT_NEAR(r[0].box_size.x(), 0.30f, 0.01f);
}

TEST_F(Tracker, DebugViewMarksTheFitAndItsVerdicts)
{
    TrackerParams params;
    params.keep_debug = true;
    const auto d      = view(depth_, k_);
    const auto out    = ObjectTracker::fit(*model_, params, params.icp_track, d, k_, truePose(), {});
    ASSERT_TRUE(out.valid);
    ASSERT_EQ(out.debug_points.size(), out.debug_inlier.size());
    int on = 0;
    for (const auto v : out.debug_inlier)
    {
        on += v == kDebugOnModel;
    }
    EXPECT_EQ(on, out.score.inliers);

    std::vector<std::uint8_t> rgb;
    int                       w = 0, h = 0;
    renderDebugImage(d, k_, { { "part", model_.get(), &out, true, {} } }, params.free_space_tol, rgb, w, h);
    EXPECT_GE(w, 640);
    EXPECT_EQ(rgb.size(), static_cast<std::size_t>(w) * h * 3);
    // The confirmed model surface is drawn in the object's colour.
    const auto c       = debugColour(0);
    long       painted = 0;
    for (std::size_t i = 0; i < rgb.size(); i += 3)
    {
        painted += rgb[i] == c[0] && rgb[i + 1] == c[1] && rgb[i + 2] == c[2];
    }
    EXPECT_GT(painted, 500);
}

TEST(PoseSpec, ParsesRegions)
{
    std::string err;
    const auto  r = parseRegion("odom 0.1 1.0 0.55 0.65 0.74 0.78 5", err);
    ASSERT_TRUE(r) << err;
    EXPECT_EQ(r->frame, "odom");
    EXPECT_TRUE(r->contains(poseFromXyzRpy(0.5f, 0.6f, 0.76f, 0, 0, 0.05f)));
    EXPECT_FALSE(r->contains(poseFromXyzRpy(0.5f, 0.6f, 0.76f, 0, 0, 0.2f)));  // turned 11 degrees
    EXPECT_FALSE(r->contains(poseFromXyzRpy(0.5f, 0.6f, 0.30f, 0, 0, 0)));     // on the floor
    EXPECT_FALSE(parseRegion("odom 1 0 0 1 0 1", err));                          // min above max
    EXPECT_FALSE(parseRegion("odom 0 1 0 1 0", err));
}

namespace
{
// Polls a MultiTracker on one image until `done` or 100 frames.
std::vector<MultiTracker::Result> run(MultiTracker& mt, const DepthImage& d, const Intrinsics& k,
                                      const std::function<bool(const std::vector<MultiTracker::Result>&)>& done)
{
    const auto lookup = [](const std::string&, double) -> std::optional<Eigen::Isometry3f> {
        return Eigen::Isometry3f::Identity();
    };
    std::vector<MultiTracker::Result> r;
    for (int f = 0; f < 100; ++f)
    {
        r = mt.update(d, k, f / 30.0, Eigen::Isometry3f::Identity(), "camera", lookup);
        if (done(r))
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return r;
}
}  // namespace

TEST_F(Tracker, AFitOutsideTheRegionIsNeverReported)
{
    ObjectConfig c;
    c.name       = "part";
    c.model_path = ::testing::TempDir() + "g1_perception_part.stl";
    c.params.global_fallback = false;
    c.priors.push_back({ "camera", truePose() });
    std::string err;
    // World = camera; the part is 0.5 m in front of it, the region 1 m.
    c.region = parseRegion("camera -1 1 -1 1 0.9 1.1", err);
    MultiTracker mt({ c });
    bool         rejected = false;
    const auto   r        = run(mt, view(depth_, k_), k_, [&](const auto& res) {
        rejected = rejected || res[0].rejected;
        EXPECT_FALSE(res[0].valid);
        return false;
    });
    EXPECT_TRUE(rejected);

    c.region = parseRegion("camera -1 1 -1 1 0.3 0.7", err);
    MultiTracker mt2({ c });
    const auto   r2 = run(mt2, view(depth_, k_), k_, [](const auto& res) { return res[0].valid; });
    EXPECT_TRUE(r2[0].valid);
}

TEST_F(Tracker, InstancesTakeDifferentParts)
{
    // A second part beside the first, both in view.
    const Eigen::Isometry3f beside = poseFromXyzRpy(0.0f, 0.15f, 0.0f, 0, 0, 0);
    std::vector<Tri>        both   = tris_;
    for (const auto& t : tris_)
    {
        both.push_back({ beside * t[0], beside * t[1], beside * t[2] });
    }
    const auto depth = render(both, truePose(), k_);

    ObjectConfig c;
    c.name       = "part";
    c.model_path = ::testing::TempDir() + "g1_perception_part.stl";
    c.params.global_fallback = false;
    // Both priors on both instances, the first part's listed first: without the instance rules
    // both would take it.
    c.priors.push_back({ "camera", truePose() });
    c.priors.push_back({ "camera", truePose() * beside });
    const auto configs = expandInstances(c, 2);
    ASSERT_EQ(configs.size(), 2u);
    EXPECT_EQ(configs[1].name, "part_2");
    MultiTracker mt(configs);
    const auto   r = run(mt, view(depth, k_), k_, [](const auto& res) { return res[0].valid && res[1].valid; });
    ASSERT_TRUE(r[0].valid);
    ASSERT_TRUE(r[1].valid);
    const double apart = (r[0].world_T_output.translation() - r[1].world_T_output.translation()).norm();
    EXPECT_NEAR(apart, 0.15, 0.003);
    EXPECT_EQ(&mt.model(0), &mt.model(1));  // loaded once
}
