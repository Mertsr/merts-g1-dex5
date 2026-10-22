#include "g1_perception/object_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <thread>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace g1_perception
{
namespace
{

// Model preparation runs while the rest of the stack is starting: the controllers, the
// simulator, move_group. Half the cores, never all of them.
int prepThreads()
{
    return static_cast<int>(std::max(1u, std::thread::hardware_concurrency() / 2));
}

std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// STL, binary or ASCII. Binary is detected by its exact size (84 + 50 * count), which an ASCII
// file starting with "solid" can still satisfy only by coincidence; exporters do write binary
// files whose header begins with "solid", so the size test comes first.
void readStl(const std::string& path, std::vector<Eigen::Vector3f>& vertices, std::vector<Eigen::Vector3i>& tris)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("cannot open " + path);
    }
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.size() >= 84)
    {
        std::uint32_t count = 0;
        std::memcpy(&count, data.data() + 80, 4);
        if (84 + 50ull * count == data.size())
        {
            vertices.reserve(3ull * count);
            for (std::uint32_t t = 0; t < count; ++t)
            {
                const char* rec = data.data() + 84 + 50ull * t + 12;  // skip the stored normal
                for (int k = 0; k < 3; ++k)
                {
                    float v[3];
                    std::memcpy(v, rec + 12 * k, 12);
                    vertices.emplace_back(v[0], v[1], v[2]);
                }
                const int b = static_cast<int>(3 * t);
                tris.emplace_back(b, b + 1, b + 2);
            }
            return;
        }
    }
    std::istringstream text(data);
    std::string        word;
    std::vector<Eigen::Vector3f> facet;
    while (text >> word)
    {
        if (word == "vertex")
        {
            float x, y, z;
            text >> x >> y >> z;
            facet.emplace_back(x, y, z);
            if (facet.size() == 3)
            {
                const int b = static_cast<int>(vertices.size());
                vertices.insert(vertices.end(), facet.begin(), facet.end());
                tris.emplace_back(b, b + 1, b + 2);
                facet.clear();
            }
        }
        else if (word == "endloop")
        {
            facet.clear();
        }
    }
    if (tris.empty())
    {
        throw std::runtime_error(path + ": no triangles (neither binary nor ASCII STL?)");
    }
}

// PLY: faces if it has them (polygons fan-triangulated), otherwise vertices only.
void readPly(const std::string& path, std::vector<Eigen::Vector3f>& vertices, std::vector<Eigen::Vector3i>& tris)
{
    pcl::PolygonMesh mesh;
    pcl::PLYReader   reader;
    if (reader.read(path, mesh) < 0)
    {
        throw std::runtime_error("cannot read PLY " + path);
    }
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);
    vertices.reserve(cloud.size());
    for (const auto& p : cloud.points)
    {
        vertices.emplace_back(p.x, p.y, p.z);
    }
    for (const auto& poly : mesh.polygons)
    {
        for (std::size_t k = 2; k < poly.vertices.size(); ++k)
        {
            tris.emplace_back(poly.vertices[0], poly.vertices[k - 1], poly.vertices[k]);
        }
    }
    if (vertices.empty())
    {
        throw std::runtime_error(path + ": PLY has no vertices");
    }
}

// Keeps, per voxel, the point nearest the voxel's centre (and its normal): an even cloud that
// stays on the surface, unlike a centroid filter which pulls points off thin features.
void voxelThin(const std::vector<Eigen::Vector3f>& pts, const std::vector<Eigen::Vector3f>& nrm, float voxel,
               std::vector<Eigen::Vector3f>& out_p, std::vector<Eigen::Vector3f>& out_n)
{
    std::unordered_map<std::int64_t, std::pair<int, float>> best;
    best.reserve(pts.size());
    for (int i = 0; i < static_cast<int>(pts.size()); ++i)
    {
        const Eigen::Vector3f q = pts[i] / voxel;
        const Eigen::Vector3f f(std::floor(q.x()), std::floor(q.y()), std::floor(q.z()));
        const std::int64_t key = (static_cast<std::int64_t>(f.x()) & 0x1FFFFF) |
                                 ((static_cast<std::int64_t>(f.y()) & 0x1FFFFF) << 21) |
                                 ((static_cast<std::int64_t>(f.z()) & 0x1FFFFF) << 42);
        const float d = (q - f - Eigen::Vector3f::Constant(0.5f)).squaredNorm();
        auto it = best.find(key);
        if (it == best.end())
        {
            best.emplace(key, std::make_pair(i, d));
        }
        else if (d < it->second.second)
        {
            it->second = { i, d };
        }
    }
    std::vector<int> keep;
    keep.reserve(best.size());
    for (const auto& kv : best)
    {
        keep.push_back(kv.second.first);
    }
    std::sort(keep.begin(), keep.end());  // deterministic order
    out_p.clear();
    out_n.clear();
    for (int i : keep)
    {
        out_p.push_back(pts[i]);
        out_n.push_back(nrm[i]);
    }
}

}  // namespace

void sampleMesh(const std::vector<Eigen::Vector3f>& v, const std::vector<Eigen::Vector3i>& tris, float spacing,
                std::vector<Eigen::Vector3f>& points, std::vector<Eigen::Vector3f>& normals)
{
    std::vector<double> cumulative;
    cumulative.reserve(tris.size());
    double total = 0.0;
    for (const auto& t : tris)
    {
        total += 0.5 * (v[t[1]] - v[t[0]]).cross(v[t[2]] - v[t[0]]).norm();
        cumulative.push_back(total);
    }
    if (total <= 0.0)
    {
        throw std::runtime_error("mesh has zero surface area");
    }
    // Enough samples that a `spacing` voxel filter afterwards sees every cell populated, capped
    // so a huge model cannot exhaust memory.
    const long n = std::min<long>(4'000'000L, static_cast<long>(4.0 * total / (spacing * spacing)) + 1000);
    std::mt19937                           rng(12345);
    std::uniform_real_distribution<double> pick(0.0, total);
    std::uniform_real_distribution<float>  unit(0.0f, 1.0f);
    points.clear();
    normals.clear();
    points.reserve(n);
    normals.reserve(n);
    for (long i = 0; i < n; ++i)
    {
        const auto idx = static_cast<std::size_t>(
            std::lower_bound(cumulative.begin(), cumulative.end(), pick(rng)) - cumulative.begin());
        const auto& t = tris[std::min(idx, tris.size() - 1)];
        float a = unit(rng), b = unit(rng);
        if (a + b > 1.0f)
        {
            a = 1.0f - a;
            b = 1.0f - b;
        }
        const Eigen::Vector3f e1 = v[t[1]] - v[t[0]], e2 = v[t[2]] - v[t[0]];
        Eigen::Vector3f       nrm = e1.cross(e2);
        const float           len = nrm.norm();
        if (len <= 0.0f)
        {
            continue;
        }
        points.push_back(v[t[0]] + a * e1 + b * e2);
        normals.push_back(nrm / len);
    }
}

std::shared_ptr<const ObjectModel> ObjectModel::load(const std::string& path, const ModelOptions& options)
{
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3i> tris;
    const std::string            ext = lower(path.substr(path.find_last_of('.') + 1));
    if (ext == "stl")
    {
        readStl(path, vertices, tris);
    }
    else if (ext == "ply")
    {
        readPly(path, vertices, tris);
    }
    else
    {
        throw std::runtime_error(path + ": unsupported model format (PLY or STL)");
    }
    for (auto& p : vertices)
    {
        p *= options.scale;
    }

    std::vector<Eigen::Vector3f> pts, nrm;
    bool                         oriented = false;
    if (!tris.empty())
    {
        sampleMesh(vertices, tris, options.voxel * 0.5f, pts, nrm);
        oriented = true;
    }
    else
    {
        // Point-only PLY (a scan): normals from the local neighbourhood, sign unknown.
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& p : vertices)
        {
            cloud->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
        }
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>                         normals;
        ne.setNumberOfThreads(static_cast<unsigned>(prepThreads()));
        ne.setInputCloud(cloud);
        ne.setKSearch(16);
        ne.compute(normals);
        for (std::size_t i = 0; i < cloud->size(); ++i)
        {
            const auto& n = normals[i];
            if (std::isfinite(n.normal_x))
            {
                pts.push_back(vertices[i]);
                nrm.emplace_back(n.normal_x, n.normal_y, n.normal_z);
            }
        }
    }
    return fromSurface(std::move(pts), std::move(nrm), oriented, options);
}

std::shared_ptr<const ObjectModel> ObjectModel::fromSurface(std::vector<Eigen::Vector3f> points,
                                                            std::vector<Eigen::Vector3f> normals,
                                                            bool normals_oriented, const ModelOptions& options)
{
    if (points.size() != normals.size() || points.size() < 50)
    {
        throw std::runtime_error("model needs at least 50 points, each with a normal");
    }
    std::shared_ptr<ObjectModel> model(new ObjectModel());
    model->options_          = options;
    model->normals_oriented_ = normals_oriented;
    voxelThin(points, normals, options.voxel, model->points_, model->normals_);
    model->prepare();
    return model;
}

void ObjectModel::prepare()
{
    bounds_.setEmpty();
    for (const auto& p : points_)
    {
        bounds_.extend(p);
    }

    // Nearest-neighbour grid over the bounds plus margin.
    const Eigen::Vector3f lo   = bounds_.min() - Eigen::Vector3f::Constant(options_.grid_margin);
    const Eigen::Vector3f span = bounds_.sizes() + Eigen::Vector3f::Constant(2 * options_.grid_margin);
    cell_                      = options_.grid_cell;
    const double volume        = static_cast<double>(span.x()) * span.y() * span.z();
    if (volume / std::pow(cell_, 3) > options_.max_cells)
    {
        cell_ = static_cast<float>(std::cbrt(volume / options_.max_cells)) * 1.001f;
    }
    grid_origin_ = lo;
    for (int k = 0; k < 3; ++k)
    {
        dims_[k] = static_cast<int>(std::ceil(span[k] / cell_)) + 1;
    }
    const long ncells = static_cast<long>(dims_[0]) * dims_[1] * dims_[2];
    grid_.assign(ncells, -1);

    // Filled by jump flooding: seed every cell holding model points with the one nearest its
    // centre, then let each cell adopt a neighbour's point if nearer, with neighbour offsets
    // halving from far to one cell (plus one extra 1-cell pass, "JFA+1"). Near-exact, and a few
    // hundred times cheaper than a kd-tree query per cell on a pallet-sized grid.
    const auto centre = [&](long ix, long iy, long iz) {
        return Eigen::Vector3f(lo.x() + (ix + 0.5f) * cell_, lo.y() + (iy + 0.5f) * cell_, lo.z() + (iz + 0.5f) * cell_);
    };
    const long sx = dims_[0], sxy = static_cast<long>(dims_[0]) * dims_[1];
    {
        std::vector<float> best(ncells, std::numeric_limits<float>::max());
        for (int i = 0; i < static_cast<int>(points_.size()); ++i)
        {
            const Eigen::Vector3f rel = (points_[i] - lo) / cell_;
            const long ix = static_cast<long>(rel.x()), iy = static_cast<long>(rel.y()), iz = static_cast<long>(rel.z());
            const long c  = iz * sxy + iy * sx + ix;
            const float d = (points_[i] - centre(ix, iy, iz)).squaredNorm();
            if (d < best[c])
            {
                best[c]  = d;
                grid_[c] = i;
            }
        }
    }
    const int reach = static_cast<int>(std::ceil(options_.grid_margin / cell_)) + 2;
    int       first = 1;
    while (first * 2 <= reach)
    {
        first *= 2;
    }
    std::vector<int> steps;
    for (int st = first; st >= 1; st /= 2)
    {
        steps.push_back(st);
    }
    steps.push_back(1);
    std::vector<int> next(ncells);
    for (const int st : steps)
    {
#pragma omp parallel for schedule(static) num_threads(prepThreads())
        for (long iz = 0; iz < dims_[2]; ++iz)
        {
            for (long iy = 0; iy < dims_[1]; ++iy)
            {
                for (long ix = 0; ix < dims_[0]; ++ix)
                {
                    const long            c  = iz * sxy + iy * sx + ix;
                    const Eigen::Vector3f ce = centre(ix, iy, iz);
                    int                   bi = grid_[c];
                    float bd = bi >= 0 ? (points_[bi] - ce).squaredNorm() : std::numeric_limits<float>::max();
                    for (int dz = -st; dz <= st; dz += st)
                    {
                        const long z = iz + dz;
                        if (z < 0 || z >= dims_[2])
                        {
                            continue;
                        }
                        for (int dy = -st; dy <= st; dy += st)
                        {
                            const long y = iy + dy;
                            if (y < 0 || y >= dims_[1])
                            {
                                continue;
                            }
                            for (int dx = -st; dx <= st; dx += st)
                            {
                                const long x = ix + dx;
                                if (x < 0 || x >= dims_[0])
                                {
                                    continue;
                                }
                                const int cand = grid_[z * sxy + y * sx + x];
                                if (cand < 0 || cand == bi)
                                {
                                    continue;
                                }
                                const float d = (points_[cand] - ce).squaredNorm();
                                if (d < bd)
                                {
                                    bd = d;
                                    bi = cand;
                                }
                            }
                        }
                    }
                    next[c] = bi;
                }
            }
        }
        grid_.swap(next);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->reserve(points_.size());
    for (const auto& p : points_)
    {
        cloud->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
    }

    // Twins across thin walls.
    twin_.assign(points_.size(), -1);
#pragma omp parallel num_threads(prepThreads())
    {
        pcl::KdTreeFLANN<pcl::PointXYZ> tree;
        tree.setInputCloud(cloud);
        std::vector<int>   idx;
        std::vector<float> d2;
#pragma omp for schedule(dynamic, 256)
        for (long i = 0; i < static_cast<long>(points_.size()); ++i)
        {
            if (tree.radiusSearch((*cloud)[i], options_.twin_radius, idx, d2) <= 0)
            {
                continue;
            }
            float best = 1e9f;
            for (std::size_t j = 0; j < idx.size(); ++j)
            {
                if (normals_[i].dot(normals_[idx[j]]) < -0.5f && d2[j] < best)
                {
                    best     = d2[j];
                    twin_[i] = idx[j];
                }
            }
        }
    }

    voxelThin(points_, normals_, options_.verify_voxel, verify_points_, verify_normals_);

    // Feature cloud for global initialisation.
    std::vector<Eigen::Vector3f> fp, fn;
    voxelThin(points_, normals_, options_.feature_voxel, fp, fn);
    feature_cloud_.reset(new NormalCloud);
    feature_cloud_->reserve(fp.size());
    for (std::size_t i = 0; i < fp.size(); ++i)
    {
        pcl::PointNormal q;
        q.x = fp[i].x(), q.y = fp[i].y(), q.z = fp[i].z();
        q.normal_x = fn[i].x(), q.normal_y = fn[i].y(), q.normal_z = fn[i].z();
        feature_cloud_->push_back(q);
    }
    features_.reset(new FeatureCloud);
    pcl::FPFHEstimationOMP<pcl::PointNormal, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
    fpfh.setNumberOfThreads(static_cast<unsigned>(prepThreads()));
    fpfh.setInputCloud(feature_cloud_);
    fpfh.setInputNormals(feature_cloud_);
    fpfh.setRadiusSearch(options_.feature_radius);
    fpfh.compute(*features_);
}

}  // namespace g1_perception
