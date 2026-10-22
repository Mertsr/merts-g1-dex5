#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "g1_perception/object_tracker_node.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<g1_perception::ObjectTrackerNode> node;
    try
    {
        node = std::make_shared<g1_perception::ObjectTrackerNode>(rclcpp::NodeOptions());
    }
    catch (const std::exception& e)
    {
        // A missing model or a malformed prior: say which, and exit non-zero so the launch shows
        // a dead process rather than a tracker that silently never publishes.
        RCLCPP_FATAL(rclcpp::get_logger("g1_object_tracker"), "cannot start: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
