#ifndef G1_PERCEPTION__DEBUG_RENDER_HPP_
#define G1_PERCEPTION__DEBUG_RENDER_HPP_

/**
 * @file debug_render.hpp
 * @brief Draws what the tracker is doing onto the depth image, for a person to watch.
 *
 *   - the depth itself in grey, near bright;
 *   - per tracked object: the region it cut out, its model's camera-facing surface at the fitted
 *     pose (in the object's colour where the depth confirms it, RED where the camera sees past
 *     it -- the evidence that would reject a wrong pose), and the fitted points that did NOT
 *     land on the model (magenta);
 *   - per searching object: every pose the prior search is trying, as boxes;
 *   - a status line per object: state, inlier ratio, coverage, violation, time.
 *
 * No ROS, no OpenCV; text comes from an embedded bitmap font.
 */

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "g1_perception/icp_tracker.hpp"

namespace g1_perception
{

struct DebugObject
{
    std::string                    name;
    const ObjectModel*             model = nullptr;
    const ObjectTracker::Output*   out   = nullptr;
    bool                           valid = false;
    std::vector<Eigen::Isometry3f> hypotheses;  ///< camera frame
};

/// The colour object `index` is drawn in, everywhere it is drawn (image, RViz markers, points).
std::array<std::uint8_t, 3> debugColour(std::size_t index);

/// Renders into `rgb` (width*height*3, rgb8, row-major). The image is the depth image scaled up
/// by an integer factor to at least 640 pixels wide, so text stays legible; `width` and `height`
/// return its size.
void renderDebugImage(const DepthImage& depth, const Intrinsics& k, const std::vector<DebugObject>& objects,
                      float free_space_tol, std::vector<std::uint8_t>& rgb, int& width, int& height);

/// Draws text at (x, y) (top-left), 7x12 pixels per character, with a dark backing box.
void drawText(std::vector<std::uint8_t>& rgb, int width, int height, int x, int y, const std::string& text,
              const std::array<std::uint8_t, 3>& colour);

}  // namespace g1_perception

#endif  // G1_PERCEPTION__DEBUG_RENDER_HPP_
