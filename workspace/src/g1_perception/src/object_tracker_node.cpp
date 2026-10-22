#include "g1_perception/object_tracker_node.hpp"

#include <sstream>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cstring>
#include <filesystem>
#include <sensor_msgs/image_encodings.hpp>
#include <stdexcept>
#include <utility>

#include "g1_perception/debug_render.hpp"

namespace g1_perception
{
namespace
{

Eigen::Isometry3f toEigen(const geometry_msgs::msg::Transform& t)
{
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    T.linear()          = Eigen::Quaternionf(static_cast<float>(t.rotation.w), static_cast<float>(t.rotation.x),
                                    static_cast<float>(t.rotation.y), static_cast<float>(t.rotation.z))
                     .normalized()
                     .toRotationMatrix();
    T.translation() = Eigen::Vector3f(static_cast<float>(t.translation.x), static_cast<float>(t.translation.y),
                                      static_cast<float>(t.translation.z));
    return T;
}

geometry_msgs::msg::Pose toPose(const Eigen::Isometry3f& T)
{
    geometry_msgs::msg::Pose p;
    p.position.x                = T.translation().x();
    p.position.y                = T.translation().y();
    p.position.z                = T.translation().z();
    const Eigen::Quaternionf q(T.linear());
    p.orientation.w = q.w();
    p.orientation.x = q.x();
    p.orientation.y = q.y();
    p.orientation.z = q.z();
    return p;
}

geometry_msgs::msg::Transform toTransform(const Eigen::Isometry3f& T)
{
    const auto                    p = toPose(T);
    geometry_msgs::msg::Transform t;
    t.translation.x = p.position.x;
    t.translation.y = p.position.y;
    t.translation.z = p.position.z;
    t.rotation      = p.orientation;
    return t;
}

// "package://<pkg>/<path>" -> the installed share path; anything else unchanged.
std::string resolvePath(const std::string& path)
{
    const std::string scheme = "package://";
    if (path.rfind(scheme, 0) != 0)
    {
        return path;
    }
    const std::string rest  = path.substr(scheme.size());
    const auto        slash = rest.find('/');
    if (slash == std::string::npos)
    {
        throw std::runtime_error("'" + path + "' names a package but no file inside it");
    }
    return ament_index_cpp::get_package_share_directory(rest.substr(0, slash)) + rest.substr(slash);
}

const char* stateName(ObjectTracker::State s) { return s == ObjectTracker::State::kTracking ? "tracking" : "searching"; }

// A part that is round about its own z (symmetric_z) gets whatever yaw ICP settled on. Keep its
// axis and turn its x to the world's +x projected across that axis (+y if the axis lies along
// x), so the reported yaw is the same every frame and from every viewpoint.
Eigen::Isometry3f facingWorldX(Eigen::Isometry3f T)
{
    const Eigen::Vector3f z = T.linear().col(2).normalized();
    Eigen::Vector3f       x = Eigen::Vector3f::UnitX() - z * z.x();
    if (x.norm() < 1e-3f)
    {
        x = Eigen::Vector3f::UnitY() - z * z.y();
    }
    x.normalize();
    T.linear() << x, z.cross(x), z;
    return T;
}

}  // namespace

ObjectTrackerNode::ObjectTrackerNode(const rclcpp::NodeOptions& options)
  : rclcpp::Node("g1_object_tracker", options)
{
    world_frame_  = declare_parameter<std::string>("world_frame", "odom");
    output_frame_ = declare_parameter<std::string>("output_frame", "camera_color_optical_frame");
    depth_scale_  = declare_parameter<double>("depth_scale", 0.001);
    tf_timeout_s_ = declare_parameter<double>("tf_timeout_s", 0.05);
    publish_tf_   = declare_parameter<bool>("publish_tf", true);
    const auto depth_topic =
        declare_parameter<std::string>("depth_topic", "/camera/aligned_depth_to_color/image_raw");
    const auto info_topic =
        declare_parameter<std::string>("info_topic", "/camera/aligned_depth_to_color/camera_info");

    const auto t0 = std::chrono::steady_clock::now();
    multi_        = std::make_unique<MultiTracker>(readObjects());
    last_verified_.resize(multi_->size());
    for (std::size_t i = 0; i < multi_->size(); ++i)
    {
        const auto& c = multi_->config(i);
        RCLCPP_INFO(get_logger(), "object '%s': %s, %zu model points, %zu priors x %zu offsets", c.name.c_str(),
                    c.model_path.c_str(), multi_->model(i).points().size(), c.priors.size(),
                    c.params.prior_offsets.size());
    }
    RCLCPP_INFO(get_logger(), "models prepared in %.1f s",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    tf_buffer_   = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    if (publish_tf_)
    {
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    // Keep-last 1: tracking wants the newest frame, never a backlog of old ones.
    const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(1);
    info_sub_             = create_subscription<sensor_msgs::msg::CameraInfo>(
        info_topic, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::SharedPtr msg) { onInfo(std::move(msg)); });
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        depth_topic, sensor_qos, [this](sensor_msgs::msg::Image::SharedPtr msg) { onDepth(std::move(msg)); });
    detections_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>("~/detections", rclcpp::SensorDataQoS());
    status_pub_     = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", rclcpp::QoS(10));
    // What the tracker sees, for a person to watch. Nothing is computed for these until something
    // subscribes, so leaving them advertised costs nothing.
    debug_image_pub_   = create_publisher<sensor_msgs::msg::Image>("~/debug/image", rclcpp::QoS(1));
    debug_markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/markers", rclcpp::QoS(1));
    debug_points_pub_  = create_publisher<sensor_msgs::msg::PointCloud2>("~/debug/points", rclcpp::QoS(1));
    depth_topic_       = depth_topic;
    info_topic_        = info_topic;
    // No frames at all is the commonest reason for a blank view: say so, once a second.
    debug_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {
        if (debug_image_pub_->get_subscription_count() == 0 ||
            std::chrono::steady_clock::now() - last_depth_ < std::chrono::seconds(2))
        {
            return;
        }
        publishDebugReason("no depth on " + depth_topic_ + " (" + std::to_string(depth_sub_->get_publisher_count()) +
                               " publishers). Simulator: sensors:=true, camera_enabled in sim_sensors.yaml",
                           nullptr);
    });

    RCLCPP_INFO(get_logger(), "tracking from %s in '%s', detections on %s in '%s'", depth_topic.c_str(),
                world_frame_.c_str(), detections_pub_->get_topic_name(), output_frame_.c_str());
}

std::vector<ObjectConfig> ObjectTrackerNode::readObjects()
{
    const auto names = declare_parameter<std::vector<std::string>>("objects", std::vector<std::string>{});
    if (names.empty())
    {
        throw std::runtime_error("parameter 'objects' is empty: nothing to track");
    }
    // Shared by every object; each can override the ones that differ per part.
    TrackerParams base;
    base.max_points     = static_cast<int>(declare_parameter<int>("tracking.max_points", base.max_points));
    base.min_ratio      = static_cast<float>(declare_parameter<double>("tracking.min_ratio", base.min_ratio));
    base.min_coverage   = static_cast<float>(declare_parameter<double>("tracking.min_coverage", base.min_coverage));
    base.max_violation  = static_cast<float>(declare_parameter<double>("tracking.max_violation", base.max_violation));
    base.lost_after     = static_cast<int>(declare_parameter<int>("tracking.lost_after", base.lost_after));
    base.global_after_s = declare_parameter<double>("tracking.global_after_s", base.global_after_s);
    base.z_max          = static_cast<float>(declare_parameter<double>("tracking.z_max", base.z_max));
    ModelOptions base_model;
    base_model.voxel = static_cast<float>(declare_parameter<double>("model.voxel", base_model.voxel));

    std::vector<ObjectConfig> out;
    for (const auto& name : names)
    {
        ObjectConfig c;
        c.name       = name;
        c.params     = base;
        c.model      = base_model;
        const auto model_param = declare_parameter<std::string>(name + ".model", "");
        c.model_path           = resolvePath(model_param);
        if (c.model_path.empty())
        {
            throw std::runtime_error("object '" + name + "' has no '" + name + ".model'");
        }
        c.model.scale = static_cast<float>(declare_parameter<double>(name + ".model_scale", 1.0));
        // RViz resolves package:// itself; anything else it needs as an absolute file:// URL.
        const std::string mesh_url = model_param.rfind("package://", 0) == 0
                                         ? model_param
                                         : "file://" + std::filesystem::absolute(c.model_path).string();
        c.params.global_fallback = declare_parameter<bool>(name + ".global_fallback", true);

        std::string error;
        for (const auto& spec : declare_parameter<std::vector<std::string>>(name + ".priors", std::vector<std::string>{}))
        {
            const auto p = parseFramedPose(spec, error);
            if (!p)
            {
                throw std::runtime_error(name + ".priors: " + error);
            }
            c.priors.push_back(*p);
        }
        c.params.prior_offsets.clear();
        for (const auto& spec :
             declare_parameter<std::vector<std::string>>(name + ".prior_offsets", std::vector<std::string>{}))
        {
            const auto p = parsePose(spec, error);
            if (!p)
            {
                throw std::runtime_error(name + ".prior_offsets: " + error);
            }
            c.params.prior_offsets.push_back(*p);
        }
        const auto sweep = declare_parameter<std::string>(name + ".prior_sweep", "");
        if (!sweep.empty())
        {
            const auto s = parseSweep(sweep, error);
            if (!s)
            {
                throw std::runtime_error(name + ".prior_sweep: " + error);
            }
            c.params.prior_offsets.insert(c.params.prior_offsets.end(), s->begin(), s->end());
        }
        if (c.params.prior_offsets.empty())
        {
            c.params.prior_offsets.push_back(Eigen::Isometry3f::Identity());
        }
        // "<metres> <radians>": every prior offset is also tried shifted by +-metres along each
        // of the model's axes and turned by +-radians about each -- a part nudged or tipped from
        // where it was put. 12 more fits per offset, so keep it for parts that get disturbed.
        const auto jitter = declare_parameter<std::string>(name + ".prior_jitter", "");
        if (!jitter.empty())
        {
            std::istringstream in(jitter);
            float              t = 0.0f, r = 0.0f;
            if (!(in >> t >> r) || t < 0.0f || r < 0.0f)
            {
                throw std::runtime_error(name + ".prior_jitter: expected '<metres> <radians>', got '" + jitter + "'");
            }
            std::vector<Eigen::Isometry3f> jittered;
            for (const auto& base : c.params.prior_offsets)
            {
                jittered.push_back(base);
                for (int axis = 0; axis < 3; ++axis)
                {
                    for (const float sign : { -1.0f, 1.0f })
                    {
                        Eigen::Isometry3f shift = Eigen::Isometry3f::Identity();
                        shift.translation()[axis] = sign * t;
                        jittered.push_back(base * shift);
                        Eigen::Isometry3f turn = Eigen::Isometry3f::Identity();
                        turn.linear() = Eigen::AngleAxisf(sign * r, Eigen::Vector3f::Unit(axis)).toRotationMatrix();
                        jittered.push_back(base * turn);
                    }
                }
            }
            c.params.prior_offsets = std::move(jittered);
        }
        const auto output = declare_parameter<std::string>(name + ".output_offset", "0 0 0");
        const auto o      = parsePose(output, error);
        if (!o)
        {
            throw std::runtime_error(name + ".output_offset: " + error);
        }
        c.model_T_output = *o;
        // "name x y z roll pitch yaw": the parser's frame slot carries the name.
        const auto named = [&](const std::string& key) {
            std::vector<FramedPose> out_poses;
            for (const auto& spec :
                 declare_parameter<std::vector<std::string>>(name + "." + key, std::vector<std::string>{}))
            {
                const auto p = parseFramedPose(spec, error);
                if (!p)
                {
                    throw std::runtime_error(name + "." + key + ": " + error);
                }
                out_poses.push_back(*p);
            }
            return out_poses;
        };
        const auto grasps = named("grasps");
        const auto slots  = named("slots");
        const bool symmetric_z = declare_parameter<bool>(name + ".symmetric_z", false);
        if (c.priors.empty() && !c.params.global_fallback)
        {
            throw std::runtime_error("object '" + name + "' has neither priors nor global_fallback: it can never be found");
        }
        const auto region = declare_parameter<std::string>(name + ".region", "");
        if (!region.empty())
        {
            c.region = parseRegion(region, error);
            if (!c.region)
            {
                throw std::runtime_error(name + ".region: " + error);
            }
        }
        c.instance_separation =
            static_cast<float>(declare_parameter<double>(name + ".instance_separation", c.instance_separation));
        // Several of the same part at once: <name>_1 .. <name>_N, sharing one model.
        const int instances = static_cast<int>(declare_parameter<int>(name + ".instances", 1));
        for (auto& inst : expandInstances(c, instances))
        {
            mesh_urls_.push_back(mesh_url);
            mesh_scales_.push_back(inst.model.scale);
            grasps_.push_back(grasps);
            slots_.push_back(slots);
            symmetric_z_.push_back(symmetric_z);
            out.push_back(std::move(inst));
        }
    }
    return out;
}

void ObjectTrackerNode::onInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    Intrinsics k;
    k.fx = static_cast<float>(msg->k[0]);
    k.cx = static_cast<float>(msg->k[2]);
    k.fy = static_cast<float>(msg->k[4]);
    k.cy = static_cast<float>(msg->k[5]);
    k.width  = static_cast<int>(msg->width);
    k.height = static_cast<int>(msg->height);
    if (k.fx <= 0.0f || k.fy <= 0.0f || k.width <= 0 || k.height <= 0)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "ignoring camera_info without intrinsics");
        return;
    }
    intrinsics_ = k;
}

std::optional<Eigen::Isometry3f> ObjectTrackerNode::lookup(const std::string& target, const std::string& source,
                                                           const rclcpp::Time& stamp, double timeout_s)
{
    try
    {
        return toEigen(tf_buffer_->lookupTransform(target, source, stamp, tf2::durationFromSec(timeout_s)).transform);
    }
    catch (const tf2::TransformException&)
    {
        return std::nullopt;
    }
}

void ObjectTrackerNode::onDepth(const sensor_msgs::msg::Image::SharedPtr msg)
{
    last_depth_ = std::chrono::steady_clock::now();
    if (!intrinsics_)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for camera_info");
        publishDebugReason("depth arrives, but no camera_info on " + info_topic_, nullptr);
        return;
    }
    const Intrinsics& k = *intrinsics_;
    DepthImage        depth;
    namespace enc = sensor_msgs::image_encodings;
    if (msg->encoding == enc::TYPE_32FC1)
    {
        depth.type = DepthImage::Type::kFloat32Metres;
    }
    else if (msg->encoding == enc::TYPE_16UC1 || msg->encoding == enc::MONO16)
    {
        depth.type      = DepthImage::Type::kUint16;
        depth.u16_scale = static_cast<float>(depth_scale_);
    }
    else
    {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "depth encoding '%s' is not 32FC1 or 16UC1",
                              msg->encoding.c_str());
        publishDebugReason("depth encoding '" + msg->encoding + "' is not 32FC1 or 16UC1", nullptr);
        return;
    }
    if (msg->is_bigendian || static_cast<int>(msg->width) != k.width || static_cast<int>(msg->height) != k.height ||
        static_cast<std::size_t>(msg->step) * msg->height > msg->data.size())
    {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                              "depth image %ux%u (step %u, %zu bytes) does not match camera_info %dx%d", msg->width,
                              msg->height, msg->step, msg->data.size(), k.width, k.height);
        publishDebugReason("depth image size does not match camera_info", nullptr);
        return;
    }
    depth.data       = msg->data.data();
    depth.width      = k.width;
    depth.height     = k.height;
    depth.step_bytes = static_cast<int>(msg->step);

    const rclcpp::Time stamp(msg->header.stamp, get_clock()->get_clock_type());
    const auto world_T_cam = lookup(world_frame_, msg->header.frame_id, stamp, tf_timeout_s_);
    if (!world_T_cam)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "no transform %s -> %s at the image stamp; skipping",
                             world_frame_.c_str(), msg->header.frame_id.c_str());
        publishDebugReason("no TF " + world_frame_ + " -> " + msg->header.frame_id + " at the image stamp", msg.get());
        return;
    }
    // The core takes seconds as a double; counted from the first frame so they keep
    // sub-microsecond resolution.
    static const std::int64_t epoch_ns = stamp.nanoseconds();
    const auto to_seconds = [](const rclcpp::Time& t) { return static_cast<double>(t.nanoseconds() - epoch_ns) * 1e-9; };
    const auto clock_type = stamp.get_clock_type();
    const auto frame_lookup = [&](const std::string& frame, double s) -> std::optional<Eigen::Isometry3f> {
        const rclcpp::Time t(epoch_ns + static_cast<std::int64_t>(s * 1e9), clock_type);
        return lookup(world_frame_, frame, t, 0.0);
    };

    multi_->setDebug(debugWanted());
    const auto t0      = std::chrono::steady_clock::now();
    const auto results = multi_->update(depth, k, to_seconds(stamp), *world_T_cam, msg->header.frame_id, frame_lookup);
    const double frame_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // The detector's contract is the camera frame; when depth arrives in a different optical
    // frame (the simulator labels its aligned depth camera_depth_optical_frame), re-express.
    Eigen::Isometry3f out_T_cam = Eigen::Isometry3f::Identity();
    if (msg->header.frame_id != output_frame_)
    {
        const auto t = lookup(output_frame_, msg->header.frame_id, stamp, tf_timeout_s_);
        if (!t)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "no transform %s -> %s; not publishing",
                                 output_frame_.c_str(), msg->header.frame_id.c_str());
            publishDebugReason("no TF " + output_frame_ + " -> " + msg->header.frame_id, msg.get());
            return;
        }
        out_T_cam = *t;
    }

    vision_msgs::msg::Detection3DArray array;
    array.header.stamp    = msg->header.stamp;
    array.header.frame_id = output_frame_;
    for (std::size_t oi = 0; oi < results.size(); ++oi)
    {
        const auto& r = results[oi];
        if (r.just_locked)
        {
            RCLCPP_INFO(get_logger(), "%s: locked (ratio %.2f, coverage %.2f, violation %.2f)", r.name.c_str(),
                        r.out.score.ratio, r.out.score.coverage, r.out.score.violation);
        }
        if (r.just_lost)
        {
            // The last rejected fit's numbers say why: low ratio/coverage with no violation is
            // occlusion or the part leaving view; violation means the fit landed off the part.
            RCLCPP_WARN(get_logger(),
                        "%s: lost, searching again (last fit: ratio %.2f, coverage %.2f, violation %.2f, %d points)",
                        r.name.c_str(), r.out.score.ratio, r.out.score.coverage, r.out.score.violation,
                        r.out.scene_points);
        }
        if (r.rejected)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "%s: dropped a fit outside its region or on another instance's part "
                                 "(ratio %.2f, coverage %.2f); searching again",
                                 r.name.c_str(), r.out.score.ratio, r.out.score.coverage);
        }
        if (r.valid && r.out.start_used > 0)
        {
            ++fallback_starts_[r.name];
        }
        if (r.valid)
        {
            ++valid_frames_[r.name];
        }
        // A frame the tracker rejects while still tracking (a hand over the part, a fit that
        // missed once) keeps reporting the last verified pose, held fixed in the world frame,
        // with score 0. Dropping the part from /objects for one frame made it blink in and out
        // for everything downstream; it leaves only when the tracker gives up and searches.
        Eigen::Isometry3f world_T_output, cam_T_output;
        float             score = r.out.score.coverage;
        if (r.valid)
        {
            world_T_output = r.world_T_output;
            cam_T_output   = r.cam_T_output;
            last_verified_[oi] = world_T_output;
        }
        else if (r.out.state == ObjectTracker::State::kTracking && last_verified_[oi])
        {
            world_T_output = *last_verified_[oi];
            cam_T_output   = world_T_cam->inverse() * world_T_output;
            score          = 0.0f;
        }
        else
        {
            last_verified_[oi].reset();
            continue;
        }
        if (symmetric_z_[oi])
        {
            world_T_output = facingWorldX(world_T_output);
            cam_T_output   = world_T_cam->inverse() * world_T_output;
        }
        const Eigen::Isometry3f pose = out_T_cam * cam_T_output;
        vision_msgs::msg::Detection3D d;
        d.header = array.header;
        d.id     = r.name;
        vision_msgs::msg::ObjectHypothesisWithPose h;
        h.hypothesis.class_id = r.name;
        h.hypothesis.score    = score;
        h.pose.pose           = toPose(pose);
        d.results.push_back(h);
        // Grasps and slots ride on the same detection, after the object's own pose, so a consumer
        // that reads results[0] sees nothing new.
        for (const auto& [kind, list] : { std::pair<const char*, const std::vector<FramedPose>*>{ "grasp", &grasps_[oi] },
                                          std::pair<const char*, const std::vector<FramedPose>*>{ "slot", &slots_[oi] } })
        {
            for (const auto& g : *list)
            {
                vision_msgs::msg::ObjectHypothesisWithPose extra;
                extra.hypothesis.class_id = r.name + "/" + kind + "/" + g.frame;
                extra.hypothesis.score    = score;
                extra.pose.pose           = toPose(pose * g.pose);
                d.results.push_back(extra);
                if (tf_broadcaster_)
                {
                    geometry_msgs::msg::TransformStamped t;
                    t.header.stamp    = msg->header.stamp;
                    t.header.frame_id = world_frame_;
                    t.child_frame_id  = "tracked/" + extra.hypothesis.class_id;
                    t.transform       = toTransform(world_T_output * g.pose);
                    tf_broadcaster_->sendTransform(t);
                }
            }
        }
        if (symmetric_z_[oi])
        {
            // Tells a consumer the part may be turned freely about its z (a place may stand a
            // can in its slot whichever way suits the hand). Its pose is the object's own.
            vision_msgs::msg::ObjectHypothesisWithPose sym;
            sym.hypothesis.class_id = r.name + "/symmetric/z";
            sym.hypothesis.score    = score;
            sym.pose.pose           = toPose(pose);
            d.results.push_back(sym);
        }
        Eigen::Isometry3f centre = pose;
        centre.translation()     = pose * r.box_centre;
        d.bbox.center            = toPose(centre);
        d.bbox.size.x            = r.box_size.x();
        d.bbox.size.y            = r.box_size.y();
        d.bbox.size.z            = r.box_size.z();
        array.detections.push_back(std::move(d));

        if (tf_broadcaster_)
        {
            geometry_msgs::msg::TransformStamped t;
            t.header.stamp    = msg->header.stamp;
            t.header.frame_id = world_frame_;
            t.child_frame_id  = "tracked/" + r.name;
            t.transform       = toTransform(world_T_output);
            tf_broadcaster_->sendTransform(t);
        }
    }
    detections_pub_->publish(array);
    publishStatus(array.header, results, frame_ms);
    if (multi_->debug())
    {
        publishDebug(*msg, depth, k, *world_T_cam, results);
    }
}

void ObjectTrackerNode::publishDebugReason(const std::string& reason, const sensor_msgs::msg::Image* msg)
{
    if (debug_image_pub_->get_subscription_count() == 0)
    {
        return;
    }
    auto img = std::make_unique<sensor_msgs::msg::Image>();
    int  w = 848, h = 480;
    // The depth itself under the message when there is a usable frame, else a blank one.
    if (msg != nullptr && intrinsics_ && msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1 &&
        static_cast<int>(msg->width) == intrinsics_->width && static_cast<int>(msg->height) == intrinsics_->height)
    {
        DepthImage d;
        d.data = msg->data.data(), d.width = intrinsics_->width, d.height = intrinsics_->height;
        d.step_bytes = static_cast<int>(msg->step);
        renderDebugImage(d, *intrinsics_, {}, 0.015f, img->data, w, h);
        img->header = msg->header;
    }
    else
    {
        img->data.assign(static_cast<std::size_t>(w) * h * 3, 30);
        img->header.stamp = now();
    }
    drawText(img->data, w, h, 8, h / 2 - 20, "TRACKER NOT RUNNING:", { 255, 60, 60 });
    // Wrapped to the image width.
    const std::size_t per_line = static_cast<std::size_t>(std::max(10, (w - 16) / 7));
    int               y        = h / 2;
    for (std::size_t i = 0; i < reason.size(); i += per_line, y += 15)
    {
        drawText(img->data, w, h, 8, y, reason.substr(i, per_line), { 255, 255, 255 });
    }
    img->width    = static_cast<std::uint32_t>(w);
    img->height   = static_cast<std::uint32_t>(h);
    img->encoding = sensor_msgs::image_encodings::RGB8;
    img->step     = static_cast<std::uint32_t>(w * 3);
    debug_image_pub_->publish(std::move(img));
}

bool ObjectTrackerNode::debugWanted() const
{
    return debug_image_pub_->get_subscription_count() > 0 || debug_markers_pub_->get_subscription_count() > 0 ||
           debug_points_pub_->get_subscription_count() > 0;
}

void ObjectTrackerNode::publishDebug(const sensor_msgs::msg::Image& msg, const DepthImage& depth, const Intrinsics& k,
                                     const Eigen::Isometry3f& world_T_cam,
                                     const std::vector<MultiTracker::Result>& results)
{
    const auto colour = [](std::size_t i, float a) {
        const auto            c = debugColour(i);
        std_msgs::msg::ColorRGBA out;
        out.r = c[0] / 255.0f, out.g = c[1] / 255.0f, out.b = c[2] / 255.0f, out.a = a;
        return out;
    };

    // 1. The depth image with the fits drawn on it.
    if (debug_image_pub_->get_subscription_count() > 0)
    {
        std::vector<DebugObject> views;
        for (std::size_t i = 0; i < results.size(); ++i)
        {
            views.push_back({ results[i].name, &multi_->model(i), &results[i].out, results[i].valid,
                              results[i].hypotheses });
        }
        auto img = std::make_unique<sensor_msgs::msg::Image>();
        int  w = 0, h = 0;
        renderDebugImage(depth, k, views, multi_->config(0).params.free_space_tol, img->data, w, h);
        img->header   = msg.header;
        img->width    = static_cast<std::uint32_t>(w);
        img->height   = static_cast<std::uint32_t>(h);
        img->encoding = sensor_msgs::image_encodings::RGB8;
        img->step     = static_cast<std::uint32_t>(w * 3);
        debug_image_pub_->publish(std::move(img));
    }

    // 2. RViz markers in the world frame: each model's mesh where it was fitted, a label, and
    //    while searching, the boxes it is trying.
    if (debug_markers_pub_->get_subscription_count() > 0)
    {
        visualization_msgs::msg::MarkerArray arr;
        visualization_msgs::msg::Marker      clear;
        clear.action = visualization_msgs::msg::Marker::DELETEALL;
        arr.markers.push_back(clear);
        int id = 0;
        for (std::size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            visualization_msgs::msg::Marker base;
            base.header.stamp    = msg.header.stamp;
            base.header.frame_id = world_frame_;
            base.ns              = r.name;
            const bool tracking  = r.out.state == ObjectTracker::State::kTracking;
            if (tracking)
            {
                const Eigen::Isometry3f world_T_model = world_T_cam * r.out.cam_T_model;
                auto                    mesh          = base;
                mesh.id                               = id++;
                mesh.type                             = visualization_msgs::msg::Marker::MESH_RESOURCE;
                mesh.mesh_resource                    = mesh_urls_[i];
                mesh.pose                             = toPose(world_T_model);
                mesh.scale.x = mesh.scale.y = mesh.scale.z = mesh_scales_[i];
                // The object's colour when verified; red when this frame's fit was rejected.
                mesh.color = colour(i, 0.75f);
                if (!r.valid)
                {
                    mesh.color.r = 1.0f, mesh.color.g = 0.15f, mesh.color.b = 0.15f;
                }
                arr.markers.push_back(mesh);

                auto label    = base;
                label.id      = id++;
                label.type    = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                label.pose    = toPose(world_T_model);
                label.pose.position.z += 0.5 * r.box_size.maxCoeff() + 0.05;
                label.scale.z = 0.04;
                label.color   = colour(i, 1.0f);
                char text[160];
                std::snprintf(text, sizeof text, "%s %s  ratio %.2f cov %.2f viol %.2f", r.name.c_str(),
                              r.valid ? "TRACK" : "REJECT", r.out.score.ratio, r.out.score.coverage,
                              r.out.score.violation);
                label.text = text;
                arr.markers.push_back(label);

                // Each grasp as the hand: the approach line (yellow) coming in along the grasp
                // frame's x, the four fingers (red, -y) and the thumb (blue, +y),
                // the palm bar (white), and a cyan cross where the hand closes.
                Eigen::Isometry3f world_T_output = world_T_model * multi_->config(i).model_T_output;
                if (symmetric_z_[i])
                {
                    world_T_output = facingWorldX(world_T_output);  // as reported on /objects
                }
                for (const auto& g : grasps_[i])
                {
                    const Eigen::Isometry3f T = world_T_output * g.pose;
                    auto lines                = base;
                    lines.id                  = id++;
                    lines.type                = visualization_msgs::msg::Marker::LINE_LIST;
                    lines.pose.orientation.w  = 1.0;
                    lines.scale.x             = 0.003;
                    const auto add = [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b, float r_, float g_,
                                         float b_) {
                        for (const auto& p : { a, b })
                        {
                            const Eigen::Vector3f w = T * p;
                            geometry_msgs::msg::Point q;
                            q.x = w.x(), q.y = w.y(), q.z = w.z();
                            lines.points.push_back(q);
                            std_msgs::msg::ColorRGBA c;
                            c.r = r_, c.g = g_, c.b = b_, c.a = 1.0f;
                            lines.colors.push_back(c);
                        }
                    };
                    // Where the RH56E2 actually is around the pinch point, in the pinch_open
                    // shape it goes down in (from the hand's kinematics, see pinch_frame in
                    // g1_description): the four fingertips 0.037 m past the point on the -y side,
                    // 0.016 m out; the thumb tip on +y, 0.021 m out. Closing brings both to it.
                    add(Eigen::Vector3f(-0.18f, 0.0f, 0.0f), Eigen::Vector3f(-0.08f, 0.0f, 0.0f), 1.0f, 0.85f, 0.1f);           // approach
                    add(Eigen::Vector3f(-0.08f, -0.016f, 0.0f), Eigen::Vector3f(-0.08f, 0.021f, 0.0f), 0.9f, 0.9f, 0.9f);  // palm
                    add(Eigen::Vector3f(-0.08f, -0.016f, 0.0f), Eigen::Vector3f(0.037f, -0.016f, 0.0f), 1.0f, 0.15f, 0.15f); // fingers
                    add(Eigen::Vector3f(-0.08f, 0.021f, 0.0f), Eigen::Vector3f(-0.009f, 0.021f, 0.0f), 0.2f, 0.4f, 1.0f);   // thumb
                    add(Eigen::Vector3f(-0.012f, 0.0f, 0.0f), Eigen::Vector3f(0.012f, 0.0f, 0.0f), 0.0f, 0.9f, 0.9f);         // cross
                    add(Eigen::Vector3f(0.0f, -0.012f, 0.0f), Eigen::Vector3f(0.0f, 0.012f, 0.0f), 0.0f, 0.9f, 0.9f);
                    add(Eigen::Vector3f(0.0f, 0.0f, -0.012f), Eigen::Vector3f(0.0f, 0.0f, 0.012f), 0.0f, 0.9f, 0.9f);
                    arr.markers.push_back(lines);

                    auto name    = label;
                    name.id      = id++;
                    name.pose    = toPose(T);
                    name.pose.position.z += 0.03;
                    name.scale.z = 0.025;
                    name.text    = g.frame;
                    arr.markers.push_back(name);
                }
                // Each slot as axes (red x, green y, blue z) and its name.
                for (const auto& sl : slots_[i])
                {
                    const Eigen::Isometry3f T = world_T_output * sl.pose;
                    auto axes                 = base;
                    axes.id                   = id++;
                    axes.type                 = visualization_msgs::msg::Marker::LINE_LIST;
                    axes.pose.orientation.w   = 1.0;
                    axes.scale.x              = 0.003;
                    for (int a = 0; a < 3; ++a)
                    {
                        for (const auto& p : { Eigen::Vector3f::Zero().eval(), (0.05f * Eigen::Vector3f::Unit(a)).eval() })
                        {
                            const Eigen::Vector3f w = T * p;
                            geometry_msgs::msg::Point q;
                            q.x = w.x(), q.y = w.y(), q.z = w.z();
                            axes.points.push_back(q);
                            std_msgs::msg::ColorRGBA c;
                            c.r = a == 0, c.g = a == 1, c.b = a == 2, c.a = 1.0f;
                            axes.colors.push_back(c);
                        }
                    }
                    arr.markers.push_back(axes);
                    auto name    = label;
                    name.id      = id++;
                    name.pose    = toPose(T);
                    name.pose.position.z += 0.06;
                    name.scale.z = 0.025;
                    name.text    = sl.frame;
                    arr.markers.push_back(name);
                }
            }
            if (!r.hypotheses.empty())
            {
                auto boxes  = base;
                boxes.id    = id++;
                boxes.type  = visualization_msgs::msg::Marker::LINE_LIST;
                boxes.pose.orientation.w = 1.0;
                boxes.scale.x            = 0.002;
                boxes.color              = colour(i, 0.5f);
                static const int edges[12][2] = { { 0, 1 }, { 1, 3 }, { 3, 2 }, { 2, 0 }, { 4, 5 }, { 5, 7 },
                                                  { 7, 6 }, { 6, 4 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } };
                const auto& bounds = multi_->model(i).bounds();
                for (const auto& hyp : r.hypotheses)
                {
                    const Eigen::Isometry3f T = world_T_cam * hyp;
                    for (const auto& e : edges)
                    {
                        for (int end = 0; end < 2; ++end)
                        {
                            const Eigen::Vector3f p =
                                T * bounds.corner(static_cast<Eigen::AlignedBox3f::CornerType>(e[end]));
                            geometry_msgs::msg::Point q;
                            q.x = p.x(), q.y = p.y(), q.z = p.z();
                            boxes.points.push_back(q);
                        }
                    }
                }
                arr.markers.push_back(boxes);
            }
        }
        debug_markers_pub_->publish(arr);
    }

    // 3. The points each fit used, in the camera frame: the object's colour on the model,
    //    magenta in its box but off it, grey outside its box.
    if (debug_points_pub_->get_subscription_count() > 0)
    {
        std::size_t n = 0;
        for (const auto& r : results)
        {
            n += r.out.debug_points.size();
        }
        auto cloud             = std::make_unique<sensor_msgs::msg::PointCloud2>();
        cloud->header          = msg.header;
        cloud->height          = 1;
        cloud->width           = static_cast<std::uint32_t>(n);
        cloud->is_bigendian    = false;
        cloud->is_dense        = true;
        cloud->point_step      = 16;
        cloud->row_step        = cloud->point_step * cloud->width;
        const char* names[4]   = { "x", "y", "z", "rgb" };
        for (std::uint32_t f = 0; f < 4; ++f)
        {
            sensor_msgs::msg::PointField field;
            field.name     = names[f];
            field.offset   = 4 * f;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count    = 1;
            cloud->fields.push_back(field);
        }
        cloud->data.resize(static_cast<std::size_t>(cloud->row_step));
        std::uint8_t* out = cloud->data.data();
        for (std::size_t i = 0; i < results.size(); ++i)
        {
            const auto&    r  = results[i];
            const auto     c  = debugColour(i);
            const std::uint32_t on  = (static_cast<std::uint32_t>(c[0]) << 16) | (c[1] << 8) | c[2];
            const std::uint32_t off = 0xff00c8, outside = 0x606060;
            for (std::size_t p = 0; p < r.out.debug_points.size(); ++p, out += 16)
            {
                std::memcpy(out, r.out.debug_points[p].data(), 12);
                const std::uint8_t v   = r.out.debug_inlier[p];
                const std::uint32_t rgb = v == kDebugOnModel ? on : (v == kDebugOffModel ? off : outside);
                std::memcpy(out + 12, &rgb, 4);
            }
        }
        debug_points_pub_->publish(std::move(cloud));
    }
}

void ObjectTrackerNode::publishStatus(const std_msgs::msg::Header& header,
                                      const std::vector<MultiTracker::Result>& results, double frame_ms)
{
    sum_ms_ += frame_ms;
    max_ms_ = std::max(max_ms_, frame_ms);
    ++frames_;
    const rclcpp::Time now = get_clock()->now();
    if (last_status_.nanoseconds() != 0 && (now - last_status_).seconds() < 0.5)
    {
        return;
    }
    const double period = last_status_.nanoseconds() != 0 ? (now - last_status_).seconds() : 0.0;
    last_status_        = now;

    diagnostic_msgs::msg::DiagnosticArray diag;
    diag.header.stamp = header.stamp;
    const auto kv     = [](const std::string& k, const std::string& v) {
        diagnostic_msgs::msg::KeyValue p;
        p.key   = k;
        p.value = v;
        return p;
    };
    diagnostic_msgs::msg::DiagnosticStatus frame;
    frame.name        = "g1_object_tracker: frame";
    frame.hardware_id = "g1_object_tracker";
    frame.level       = diagnostic_msgs::msg::DiagnosticStatus::OK;
    frame.message     = "ok";
    frame.values.push_back(kv("mean_ms", std::to_string(sum_ms_ / frames_)));
    frame.values.push_back(kv("max_ms", std::to_string(max_ms_)));
    if (period > 0.0)
    {
        frame.values.push_back(kv("input_hz", std::to_string(frames_ / period)));
    }
    diag.status.push_back(frame);
    sum_ms_ = max_ms_ = 0.0;
    frames_           = 0;

    for (const auto& r : results)
    {
        diagnostic_msgs::msg::DiagnosticStatus s;
        s.name        = "g1_object_tracker: " + r.name;
        s.hardware_id = "g1_object_tracker";
        const bool tracking = r.out.state == ObjectTracker::State::kTracking;
        s.level   = r.valid ? diagnostic_msgs::msg::DiagnosticStatus::OK : diagnostic_msgs::msg::DiagnosticStatus::WARN;
        s.message = r.valid ? "tracking" : (tracking ? "tracking, this frame rejected" : "searching");
        s.values.push_back(kv("state", stateName(r.out.state)));
        s.values.push_back(kv("inlier_ratio", std::to_string(r.out.score.ratio)));
        s.values.push_back(kv("coverage", std::to_string(r.out.score.coverage)));
        s.values.push_back(kv("violation", std::to_string(r.out.score.violation)));
        s.values.push_back(kv("points", std::to_string(r.out.scene_points)));
        s.values.push_back(kv("icp_iterations", std::to_string(r.out.icp.iterations)));
        s.values.push_back(kv("ms", std::to_string(r.out.ms)));
        // Frames that needed a start other than the TF prediction. Many of them, with the part
        // still, means TF (odometry, joint states, stamps) disagrees with what the camera sees.
        s.values.push_back(kv("fallback_starts", std::to_string(fallback_starts_[r.name]) + "/" +
                                                     std::to_string(valid_frames_[r.name])));
        if (valid_frames_[r.name] >= 30 && fallback_starts_[r.name] * 5 > valid_frames_[r.name])
        {
            RCLCPP_WARN(get_logger(),
                        "%s: %d of the last %d tracked frames could not be predicted from TF; check that TF "
                        "(odom and the waist/head joints) is current at the depth image's stamp",
                        r.name.c_str(), fallback_starts_[r.name], valid_frames_[r.name]);
        }
        fallback_starts_[r.name] = valid_frames_[r.name] = 0;
        diag.status.push_back(s);
    }
    status_pub_->publish(diag);
}

}  // namespace g1_perception
