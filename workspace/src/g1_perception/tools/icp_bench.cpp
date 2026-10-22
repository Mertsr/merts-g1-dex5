// Offline benchmark: replays a recorded sequence (depth frames, camera poses, ground-truth object
// poses) through MultiTracker -- the same code the ROS node runs, with this file standing in for
// TF -- and reports pose error against ground truth and time per frame. No ROS.
//
//   icp_bench <sequence_dir> <object>=<model_file>[@scale] [<object>=<model_file>[@scale] ...]
//
// <sequence_dir> holds:
//   meta.txt          "width height fx fy cx cy frames"
//   cam.txt           world_T_camera_optical per frame, 12 numbers (3x4 row-major) per line
//   gt_<object>.txt   world_T_model per frame, same format (grading only, never shown to tracking)
//   pose_<frame>.txt  optional extra frames priors can hang off (a hand), same format
//   priors.txt        optional: "prior <object> <world|object|frame> <12 numbers>" and
//                     "offset <object> <12 numbers>" lines
//   depth/NNNNN.bin   float32 metres, width*height
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "g1_perception/debug_render.hpp"
#include "g1_perception/multi_tracker.hpp"

using namespace g1_perception;

namespace
{

bool readPose(std::istream& in, Eigen::Isometry3f& T)
{
    T = Eigen::Isometry3f::Identity();
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            if (!(in >> T.matrix()(r, c)))
            {
                return false;
            }
        }
    }
    return true;
}

std::vector<Eigen::Isometry3f> readPoses(const std::string& path)
{
    std::ifstream                  in(path);
    std::vector<Eigen::Isometry3f> out;
    Eigen::Isometry3f              T;
    while (readPose(in, T))
    {
        out.push_back(T);
    }
    return out;
}

struct Stats
{
    int    tracked = 0, first = -1, lost = 0;
    double sum_ms = 0, max_ms = 0, sum_t = 0, max_t = 0, sum_r = 0, max_r = 0;
    std::vector<double> ms;
};

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: icp_bench <sequence_dir> <object>=<model>[@scale] ...\n");
        return 2;
    }
    const std::string dir = argv[1];
    std::ifstream     meta(dir + "/meta.txt");
    Intrinsics        k;
    int               frames = 0;
    meta >> k.width >> k.height >> k.fx >> k.fy >> k.cx >> k.cy >> frames;
    const auto cams = readPoses(dir + "/cam.txt");

    std::vector<ObjectConfig>                             configs;
    std::map<std::string, std::vector<Eigen::Isometry3f>> gts, extra_frames;
    for (int a = 2; a < argc; ++a)
    {
        const std::string spec = argv[a];
        const auto        eq   = spec.find('=');
        ObjectConfig      c;
        c.name       = spec.substr(0, eq);
        c.model_path = spec.substr(eq + 1);
        if (const auto at = c.model_path.find('@'); at != std::string::npos)
        {
            c.model.scale = std::stof(c.model_path.substr(at + 1));
            c.model_path  = c.model_path.substr(0, at);
        }
        c.params.prior_offsets.clear();
        gts[c.name] = readPoses(dir + "/gt_" + c.name + ".txt");
        configs.push_back(c);
    }
    {
        std::ifstream in(dir + "/priors.txt");
        std::string   kind, obj;
        while (in >> kind >> obj)
        {
            std::string       parent;
            Eigen::Isometry3f T;
            if (kind == "prior")
            {
                in >> parent;
            }
            if (!readPose(in, T))
            {
                break;
            }
            for (auto& c : configs)
            {
                if (c.name != obj)
                {
                    continue;
                }
                if (kind == "prior")
                {
                    c.priors.push_back({ parent, T });
                    const bool is_object = std::any_of(configs.begin(), configs.end(),
                                                       [&](const ObjectConfig& o) { return o.name == parent; });
                    if (parent != "world" && !is_object && !extra_frames.count(parent))
                    {
                        extra_frames[parent] = readPoses(dir + "/pose_" + parent + ".txt");
                    }
                }
                else
                {
                    c.params.prior_offsets.push_back(T);
                }
            }
        }
    }
    for (auto& c : configs)
    {
        if (c.params.prior_offsets.empty())
        {
            c.params.prior_offsets.push_back(Eigen::Isometry3f::Identity());
        }
    }
    const auto        t0 = std::chrono::steady_clock::now();
    MultiTracker      mt(configs);
    std::printf("models prepared in %.0f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

    // Robustness knobs, all off by default. They make the bench's TF worse than the truth the
    // way a robot's is: TF_ERR_MM / TF_ERR_MRAD add white noise to the camera pose the tracker is
    // given (odometry jitter), TF_LAG_FRAMES hands it the camera pose from that many frames ago
    // (TF and image stamps out of step), STAMP_JITTER_MS jitters the image stamps.
    const auto envf = [](const char* n) { const char* v = std::getenv(n); return v ? std::atof(v) : 0.0; };
    const double tf_err_m = envf("TF_ERR_MM") * 1e-3, tf_err_rad = envf("TF_ERR_MRAD") * 1e-3;
    const int    tf_lag   = static_cast<int>(envf("TF_LAG_FRAMES"));
    const double jitter_s = envf("STAMP_JITTER_MS") * 1e-3;
    std::vector<Eigen::Isometry3f> tf_cams(cams.size());
    std::vector<double>            stamps(frames);
    {
        std::mt19937                          rng(7);
        std::normal_distribution<float>       n01(0.0f, 1.0f);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (std::size_t i = 0; i < cams.size(); ++i)
        {
            const Eigen::Vector3f w(n01(rng), n01(rng), n01(rng));
            Eigen::Isometry3f     e = Eigen::Isometry3f::Identity();
            e.linear()              = Eigen::AngleAxisf(static_cast<float>(tf_err_rad) * w.norm(), w.normalized()).toRotationMatrix();
            e.translation()         = Eigen::Vector3f(n01(rng), n01(rng), n01(rng)) * static_cast<float>(tf_err_m);
            tf_cams[i]              = cams[std::max<int>(0, static_cast<int>(i) - tf_lag)] * e;
        }
        for (int f = 0; f < frames; ++f)
        {
            stamps[f] = f / 30.0 + jitter_s * u(rng);
        }
    }

    // The bench's TF: the camera and any extra frame, by frame index at 30 Hz.
    const auto lookup = [&](const std::string& frame, double stamp) -> std::optional<Eigen::Isometry3f> {
        const int fi = std::max(0, std::min(frames - 1, static_cast<int>(stamp * 30.0 + 0.5)));
        if (frame == "world")
        {
            return Eigen::Isometry3f::Identity();
        }
        if (frame == "camera")
        {
            return tf_cams[fi];
        }
        const auto it = extra_frames.find(frame);
        if (it != extra_frames.end() && fi < static_cast<int>(it->second.size()))
        {
            return it->second[fi];
        }
        return std::nullopt;
    };

    // DEBUG_DIR=<dir>: also write the debug view (what the node publishes on ~/debug/image)
    // every DEBUG_EVERY frames (default 10) as <dir>/NNNNN.ppm.
    const char* debug_dir   = std::getenv("DEBUG_DIR");
    const int   debug_every = std::getenv("DEBUG_EVERY") ? std::max(1, static_cast<int>(envf("DEBUG_EVERY"))) : 10;
    mt.setDebug(debug_dir != nullptr);

    std::vector<float>           buf(static_cast<std::size_t>(k.width) * k.height);
    std::map<std::string, Stats> stats;
    double                       sum_frame_ms = 0, max_frame_ms = 0;
    for (int f = 0; f < frames; ++f)
    {
        char fname[64];
        std::snprintf(fname, sizeof fname, "/depth/%05d.bin", f);
        std::ifstream in(dir + fname, std::ios::binary);
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size() * 4));
        DepthImage d;
        d.data = buf.data(), d.width = k.width, d.height = k.height, d.step_bytes = k.width * 4;

        const auto ft0     = std::chrono::steady_clock::now();
        // Ground truth is never shown to the tracker; it only grades the answer.
        const auto results = mt.update(d, k, stamps[f], tf_cams[f], "camera", lookup);
        const double fms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ft0).count();
        sum_frame_ms += fms, max_frame_ms = std::max(max_frame_ms, fms);
        if (debug_dir != nullptr && f % debug_every == 0)
        {
            std::vector<DebugObject> views;
            for (std::size_t i = 0; i < results.size(); ++i)
            {
                views.push_back({ results[i].name, &mt.model(i), &results[i].out, results[i].valid, results[i].hypotheses });
            }
            std::vector<std::uint8_t> rgb;
            int                       w = 0, h = 0;
            renderDebugImage(d, k, views, mt.config(0).params.free_space_tol, rgb, w, h);
            char path[512];
            std::snprintf(path, sizeof path, "%s/%05d.ppm", debug_dir, f);
            std::ofstream ppm(path, std::ios::binary);
            ppm << "P6\n" << w << " " << h << "\n255\n";
            ppm.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
        }
        bool searching = false;
        for (std::size_t i = 0; i < results.size(); ++i)
        {
            const auto& r   = results[i];
            Stats&      st  = stats[r.name];
            const auto& out = r.out;
            if (r.valid)
            {
                // Graded in the model frame, which is what gt_<object>.txt holds.
                const Eigen::Isometry3f world_T_model = r.world_T_output * mt.config(i).model_T_output.inverse();
                if (st.first < 0)
                {
                    st.first = f;
                }
                const Eigen::Isometry3f err = gts[r.name][f].inverse() * world_T_model;
                const double            et  = err.translation().norm() * 1000.0;
                const double            er  = Eigen::AngleAxisf(err.linear()).angle() * 180.0 / M_PI;
                ++st.tracked;
                st.ms.push_back(out.ms);
                st.sum_ms += out.ms, st.max_ms = std::max(st.max_ms, out.ms);
                st.sum_t += et, st.max_t = std::max(st.max_t, et);
                st.sum_r += er, st.max_r = std::max(st.max_r, er);
                if (f % 30 == 0 || et > 5.0 || std::getenv("ALL"))
                {
                    std::printf("  f%03d %-7s err %6.2f mm %5.2f deg  %5.2f ms  %4d pts  ratio %.2f cov %.2f viol %.2f  it %d\n",
                                f, r.name.c_str(), et, er, out.ms, out.scene_points, out.score.ratio,
                                out.score.coverage, out.score.violation, out.icp.iterations);
                }
            }
            else if (st.first >= 0)
            {
                ++st.lost;
                std::printf("  f%03d %-7s NOT VALID (state %d, ratio %.2f cov %.2f viol %.2f)\n", f, r.name.c_str(),
                            static_cast<int>(out.state), out.score.ratio, out.score.coverage, out.score.violation);
            }
            searching = searching || out.state == ObjectTracker::State::kSearching;
        }
        if (searching)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(33));  // real time: let the search run
        }
    }

    int rc = 0;
    for (std::size_t i = 0; i < mt.size(); ++i)
    {
        const std::string& name = mt.config(i).name;
        Stats&             st   = stats[name];
        if (st.tracked == 0)
        {
            std::printf("RESULT %s: never tracked\n", name.c_str());
            rc = 1;
            continue;
        }
        std::sort(st.ms.begin(), st.ms.end());
        const double p95 = st.ms[std::min(st.ms.size() - 1, static_cast<std::size_t>(st.ms.size() * 0.95))];
        std::printf("RESULT %s: locked at frame %d, valid %d/%d | error mean %.2f mm / %.2f deg, max %.2f mm / %.2f deg"
                    " | time mean %.2f ms, p95 %.2f ms, max %.2f ms\n",
                    name.c_str(), st.first, st.tracked, frames - st.first, st.sum_t / st.tracked,
                    st.sum_r / st.tracked, st.max_t, st.max_r, st.sum_ms / st.tracked, p95, st.max_ms);
    }
    std::printf("RESULT frame: all objects %.2f ms mean, %.2f ms max (%.0f fps)\n", sum_frame_ms / frames, max_frame_ms,
                1000.0 * frames / sum_frame_ms);
    return rc;
}
