#ifndef G1_PERCEPTION__OBJECT_MODEL_HPP_
#define G1_PERCEPTION__OBJECT_MODEL_HPP_

/**
 * @file object_model.hpp
 * @brief A rigid object's 3D model, prepared once for fast tracking.
 *
 * Loaded from a mesh (PLY or STL; faces are sampled into a surface cloud with exact normals) or
 * a point-only PLY (normals estimated). Two things are precomputed so that tracking never builds
 * a search structure per frame:
 *
 *   - a dense nearest-neighbour grid around the model: every cell stores the index of the
 *     closest model point, so a correspondence lookup is one array read instead of a kd-tree
 *     query. This is what makes 30+ fps ICP cheap -- the model never moves in its own frame, so
 *     its nearest-point field can be computed once;
 *   - an FPFH feature cloud for global (re)initialisation.
 *
 * No ROS. Units are metres; a model authored in millimetres is scaled on load.
 */

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace g1_perception
{

struct ModelOptions
{
    /// Spacing of the tracking cloud. Finer is more precise and slower to score, not to track.
    float voxel = 0.003f;
    /// Nearest-neighbour grid cell. Enlarged automatically if the grid would exceed max_cells.
    float grid_cell = 0.002f;
    /// How far past the model's bounds the grid reaches. Must cover the largest correspondence
    /// distance ICP is allowed to use.
    float grid_margin = 0.03f;
    long max_cells = 6'000'000;
    /// Spacing of the cloud FPFH features are computed on (global initialisation only).
    float feature_voxel = 0.008f;
    float feature_normal_radius = 0.02f;
    float feature_radius = 0.04f;
    /// Multiply every coordinate by this on load: 0.001 for a model authored in millimetres.
    float scale = 1.0f;
    /// Thin walls: a point's "twin" is the nearest point on the wall's other face within this
    /// distance. With back-face rejection, a scene point whose nearest model point is on the far
    /// face is matched to the twin instead of being dropped -- on a 2 mm plastic shell the far
    /// face is often the nearer one.
    float twin_radius = 0.006f;
    /// Spacing of the sparse cloud used to check a pose against the depth image.
    float verify_voxel = 0.01f;
};

class ObjectModel
{
public:
    using FeatureCloud = pcl::PointCloud<pcl::FPFHSignature33>;
    using NormalCloud  = pcl::PointCloud<pcl::PointNormal>;

    /// Loads and prepares a model. Throws std::runtime_error with the reason on any failure.
    static std::shared_ptr<const ObjectModel> load(const std::string& path, const ModelOptions& options);

    /// Builds from points + normals already in metres (tests, synthetic models).
    static std::shared_ptr<const ObjectModel> fromSurface(
        std::vector<Eigen::Vector3f> points, std::vector<Eigen::Vector3f> normals,
        bool normals_oriented, const ModelOptions& options);

    const std::vector<Eigen::Vector3f>& points() const { return points_; }
    const std::vector<Eigen::Vector3f>& normals() const { return normals_; }
    /// True when normals come from consistently wound mesh faces and point outwards, which is
    /// what back-face rejection needs. Estimated normals have an arbitrary sign.
    bool normalsOriented() const { return normals_oriented_; }
    const Eigen::AlignedBox3f& bounds() const { return bounds_; }
    float gridCell() const { return cell_; }

    /// Index of the model point nearest to `p` (model frame), or -1 outside the grid.
    int nearest(const Eigen::Vector3f& p) const
    {
        const Eigen::Vector3f rel = (p - grid_origin_) / cell_;
        const int ix = static_cast<int>(rel.x());
        const int iy = static_cast<int>(rel.y());
        const int iz = static_cast<int>(rel.z());
        if (rel.x() < 0 || rel.y() < 0 || rel.z() < 0 || ix >= dims_[0] || iy >= dims_[1] || iz >= dims_[2])
        {
            return -1;
        }
        return grid_[(static_cast<long>(iz) * dims_[1] + iy) * dims_[0] + ix];
    }

    /// Sparse surface samples (with normals) for checking a pose against the depth image.
    const std::vector<Eigen::Vector3f>& verifyPoints() const { return verify_points_; }
    const std::vector<Eigen::Vector3f>& verifyNormals() const { return verify_normals_; }

    /// The point on the opposite face of a thin wall (see ModelOptions::twin_radius), or -1.
    int twin(int i) const { return twin_[i]; }

    NormalCloud::ConstPtr featureCloud() const { return feature_cloud_; }
    FeatureCloud::ConstPtr features() const { return features_; }
    const ModelOptions& options() const { return options_; }

private:
    ObjectModel() = default;
    void prepare();

    ModelOptions                 options_;
    std::vector<Eigen::Vector3f> points_;
    std::vector<Eigen::Vector3f> normals_;
    bool                         normals_oriented_ = false;
    Eigen::AlignedBox3f          bounds_;

    Eigen::Vector3f  grid_origin_ = Eigen::Vector3f::Zero();
    float            cell_        = 0.0f;
    int              dims_[3]     = { 0, 0, 0 };
    std::vector<int> grid_;
    std::vector<int> twin_;
    std::vector<Eigen::Vector3f> verify_points_, verify_normals_;

    NormalCloud::Ptr  feature_cloud_;
    FeatureCloud::Ptr features_;
};

/// Area-weighted surface sampling of a triangle mesh, `spacing` metres apart on average, with
/// each sample carrying its triangle's normal. Deterministic (fixed seed).
void sampleMesh(
    const std::vector<Eigen::Vector3f>& vertices, const std::vector<Eigen::Vector3i>& triangles,
    float spacing, std::vector<Eigen::Vector3f>& points, std::vector<Eigen::Vector3f>& normals);

}  // namespace g1_perception

#endif  // G1_PERCEPTION__OBJECT_MODEL_HPP_
