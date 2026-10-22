/**
 * @file depth_cloud_main.cpp
 * @brief The head depth camera as a sparse PointCloud2, for MoveIt's octomap.
 *
 * The octomap is fed by the Mid360 on the head, which does not see what is right below and in
 * front of the hands -- a tray edge, the parts beside the one being picked. The depth camera
 * looks exactly there. MoveIt's own depth-image updater cannot be used (its YAML never applies on
 * Humble 2.5.9, see g1_moveit_config/config/sensors_3d.yaml), and depth_image_proc's subscription
 * does not match the relay's best-effort QoS, so this converts it: every `stride`-th pixel, valid
 * depth in [min_range, max_range], in the depth image's own frame, best-effort like the LiDAR.
 *
 * 32FC1 is metres; 16UC1 is multiplied by depth_scale (a real RealSense: 0.001).
 */

#include <cmath>
#include <cstring>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <string>

namespace
{

class DepthCloud : public rclcpp::Node
{
public:
    DepthCloud() : rclcpp::Node("g1_depth_cloud")
    {
        stride_      = std::max(1, static_cast<int>(declare_parameter<int>("stride", 4)));
        min_range_   = static_cast<float>(declare_parameter<double>("min_range", 0.15));
        max_range_   = static_cast<float>(declare_parameter<double>("max_range", 2.0));
        depth_scale_ = static_cast<float>(declare_parameter<double>("depth_scale", 0.001));
        const double rate = declare_parameter<double>("max_rate_hz", 10.0);
        min_period_       = rate > 0.0 ? 1.0 / rate : 0.0;

        const auto depth_topic = declare_parameter<std::string>("depth_topic", "/camera/depth_half/image_raw");
        const auto info_topic  = declare_parameter<std::string>("info_topic", "/camera/depth_half/camera_info");
        pub_  = create_publisher<sensor_msgs::msg::PointCloud2>("~/points", rclcpp::SensorDataQoS().keep_last(1));
        info_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            info_topic, rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::CameraInfo::SharedPtr m) { info_msg_ = *m; });
        depth_ = create_subscription<sensor_msgs::msg::Image>(
            depth_topic, rclcpp::SensorDataQoS().keep_last(1),
            [this](sensor_msgs::msg::Image::SharedPtr m) { onDepth(*m); });
        RCLCPP_INFO(get_logger(), "%s -> %s, every %d px, %.2f..%.2f m", depth_topic.c_str(), pub_->get_topic_name(),
                    stride_, min_range_, max_range_);
    }

private:
    void onDepth(const sensor_msgs::msg::Image& img)
    {
        if (!info_msg_ || pub_->get_subscription_count() == 0)
        {
            return;
        }
        const double t = rclcpp::Time(img.header.stamp).seconds();
        if (t - last_ < min_period_ && t >= last_)
        {
            return;
        }
        last_ = t;
        const bool is_float = img.encoding == "32FC1";
        if (!is_float && img.encoding != "16UC1")
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "unsupported depth encoding '%s'",
                                 img.encoding.c_str());
            return;
        }
        // Intrinsics scaled to this image, in case camera_info is for another resolution.
        const auto&  k  = info_msg_->k;
        const double sx = info_msg_->width ? static_cast<double>(img.width) / info_msg_->width : 1.0;
        const double sy = info_msg_->height ? static_cast<double>(img.height) / info_msg_->height : 1.0;
        const float  fx = static_cast<float>(k[0] * sx), cx = static_cast<float>(k[2] * sx);
        const float  fy = static_cast<float>(k[4] * sy), cy = static_cast<float>(k[5] * sy);
        if (fx <= 0.0f || fy <= 0.0f)
        {
            return;
        }

        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header = img.header;
        cloud.height = 1;
        cloud.is_dense = true;
        cloud.is_bigendian = false;
        cloud.point_step   = 12;
        for (const char* name : { "x", "y", "z" })
        {
            sensor_msgs::msg::PointField f;
            f.name     = name;
            f.offset   = static_cast<uint32_t>(cloud.fields.size() * 4);
            f.datatype = sensor_msgs::msg::PointField::FLOAT32;
            f.count    = 1;
            cloud.fields.push_back(f);
        }
        cloud.data.reserve(static_cast<std::size_t>(img.width / stride_ + 1) * (img.height / stride_ + 1) * 12);
        for (uint32_t v = 0; v < img.height; v += stride_)
        {
            const uint8_t* row = img.data.data() + static_cast<std::size_t>(v) * img.step;
            for (uint32_t u = 0; u < img.width; u += stride_)
            {
                float z;
                if (is_float)
                {
                    std::memcpy(&z, row + u * 4, 4);
                }
                else
                {
                    uint16_t raw;
                    std::memcpy(&raw, row + u * 2, 2);
                    z = raw * depth_scale_;
                }
                if (!std::isfinite(z) || z < min_range_ || z > max_range_)
                {
                    continue;
                }
                const float p[3] = { (u - cx) / fx * z, (v - cy) / fy * z, z };
                const auto* b    = reinterpret_cast<const uint8_t*>(p);
                cloud.data.insert(cloud.data.end(), b, b + 12);
            }
        }
        cloud.width    = static_cast<uint32_t>(cloud.data.size() / 12);
        cloud.row_step = static_cast<uint32_t>(cloud.data.size());
        pub_->publish(cloud);
    }

    int                                                         stride_;
    float                                                       min_range_, max_range_, depth_scale_;
    double                                                      min_period_, last_ = -1e9;
    std::optional<sensor_msgs::msg::CameraInfo>                 info_msg_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr     depth_;
};

}  // namespace

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthCloud>());
    rclcpp::shutdown();
    return 0;
}
