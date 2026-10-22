#ifndef G1_PERCEPTION__OBJECT_TRACKER_NODE_HPP_
#define G1_PERCEPTION__OBJECT_TRACKER_NODE_HPP_

/**
 * @file object_tracker_node.hpp
 * @brief Model-based 6D tracking of known parts from the head depth camera, at camera rate.
 *
 * Depth in, one Detection3DArray per frame out, in the camera frame and at the image's stamp --
 * the contract g1_object_pose_source's `depth_tracker` source consumes, so the skills read
 * these poses from /objects exactly as they read simulator ground truth.
 *
 * All the tracking is g1_perception::MultiTracker; this node converts messages and answers its
 * questions about frames from TF. See README.md for the configuration.
 */

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <chrono>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <map>
#include <memory>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "g1_perception/multi_tracker.hpp"

namespace g1_perception
{

class ObjectTrackerNode : public rclcpp::Node
{
public:
    explicit ObjectTrackerNode(const rclcpp::NodeOptions& options);

private:
    std::vector<ObjectConfig>        readObjects();
    void                             onInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void                             onDepth(const sensor_msgs::msg::Image::SharedPtr msg);
    std::optional<Eigen::Isometry3f> lookup(const std::string& target, const std::string& source,
                                            const rclcpp::Time& stamp, double timeout_s);
    /// The three debug views (see README "Watching it work"). Built only while someone subscribes.
    bool debugWanted() const;
    /// When the tracker cannot run on a frame (no camera_info, no TF, a bad encoding) or gets no
    /// frames at all, the debug image says so instead of staying blank.
    void publishDebugReason(const std::string& reason, const sensor_msgs::msg::Image* msg);
    void publishDebug(const sensor_msgs::msg::Image& msg, const DepthImage& depth, const Intrinsics& k,
                      const Eigen::Isometry3f& world_T_cam, const std::vector<MultiTracker::Result>& results);
    void publishStatus(const std_msgs::msg::Header& header, const std::vector<MultiTracker::Result>& results,
                       double frame_ms);

    std::string world_frame_;
    std::string output_frame_;
    double      depth_scale_;
    double      tf_timeout_s_;
    bool        publish_tf_;

    std::unique_ptr<MultiTracker>                multi_;
    std::optional<Intrinsics>                    intrinsics_;
    std::unique_ptr<tf2_ros::Buffer>             tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>  tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr           depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr      info_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr   detections_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr status_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr               debug_image_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr  debug_markers_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr         debug_points_pub_;
    rclcpp::TimerBase::SharedPtr debug_timer_;
    std::string                  depth_topic_, info_topic_;
    std::chrono::steady_clock::time_point last_depth_{};
    /// Per object, named poses in its OUTPUT frame: where a hand takes it (grasps) and where
    /// another object sits on it (slots). Published as extra hypotheses on each detection,
    /// "<object>/grasp/<name>" and "<object>/slot/<name>", so they travel through /objects with
    /// the object's own pose and stamp.
    std::vector<std::vector<FramedPose>> grasps_, slots_;
    /// Per instance: the part is round about its own z (a can, a bottle), so the yaw a fit
    /// reports is meaningless. Its pose is reported with x turned to face the world frame's +x
    /// instead, so its yaw is the same every frame and from every view, and the grasps and slots
    /// that ride on it stay put. (<name>.symmetric_z)
    std::vector<bool> symmetric_z_;
    /// Per object, the last verified pose (world frame), reported through rejected frames while
    /// the object is still being tracked.
    std::vector<std::optional<Eigen::Isometry3f>> last_verified_;
    std::vector<std::string> mesh_urls_;    ///< per object, for RViz mesh markers
    std::vector<double>      mesh_scales_;
    rclcpp::Time                                                       last_status_{ 0, 0, RCL_ROS_TIME };
    // Frame timing over the last status period, for the log line and the diagnostics.
    double sum_ms_ = 0.0, max_ms_ = 0.0;
    int    frames_ = 0;
    std::map<std::string, int> fallback_starts_, valid_frames_;
};

}  // namespace g1_perception

#endif  // G1_PERCEPTION__OBJECT_TRACKER_NODE_HPP_
