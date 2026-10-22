#include "g1_perception/debug_render.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace g1_perception
{
namespace
{

#include "debug_font.inc"

using Rgb = std::array<std::uint8_t, 3>;

const Rgb kRed{ 255, 40, 40 };
const Rgb kMagenta{ 255, 0, 200 };
const Rgb kWhite{ 235, 235, 235 };

struct Canvas
{
    std::vector<std::uint8_t>& px;
    int                        w, h;

    void set(int x, int y, const Rgb& c)
    {
        if (x < 0 || y < 0 || x >= w || y >= h)
        {
            return;
        }
        std::uint8_t* p = &px[(static_cast<std::size_t>(y) * w + x) * 3];
        p[0] = c[0], p[1] = c[1], p[2] = c[2];
    }
    void dot(int x, int y, const Rgb& c, int r)
    {
        for (int dy = 0; dy < r; ++dy)
        {
            for (int dx = 0; dx < r; ++dx)
            {
                set(x + dx, y + dy, c);
            }
        }
    }
    void darken(int x0, int y0, int x1, int y1)
    {
        for (int y = std::max(0, y0); y < std::min(h, y1); ++y)
        {
            for (int x = std::max(0, x0); x < std::min(w, x1); ++x)
            {
                std::uint8_t* p = &px[(static_cast<std::size_t>(y) * w + x) * 3];
                p[0] /= 4, p[1] /= 4, p[2] /= 4;
            }
        }
    }
    void line(float x0, float y0, float x1, float y1, const Rgb& c, int dash = 0)
    {
        const int n = static_cast<int>(std::max(std::abs(x1 - x0), std::abs(y1 - y0))) + 1;
        if (n > 20000)
        {
            return;  // a point behind the camera projected to infinity
        }
        for (int i = 0; i <= n; ++i)
        {
            if (dash > 0 && (i / dash) % 2 == 1)
            {
                continue;
            }
            const float t = static_cast<float>(i) / n;
            set(static_cast<int>(x0 + (x1 - x0) * t), static_cast<int>(y0 + (y1 - y0) * t), c);
        }
    }
    void rect(int x0, int y0, int x1, int y1, const Rgb& c, int dash = 0)
    {
        line(x0, y0, x1, y0, c, dash);
        line(x1, y0, x1, y1, c, dash);
        line(x1, y1, x0, y1, c, dash);
        line(x0, y1, x0, y0, c, dash);
    }
};

bool project(const Eigen::Vector3f& p, const Intrinsics& k, float& u, float& v)
{
    if (p.z() <= 0.05f)
    {
        return false;
    }
    u = k.fx * p.x() / p.z() + k.cx;
    v = k.fy * p.y() / p.z() + k.cy;
    return true;
}

void drawBox(Canvas& c, const Eigen::AlignedBox3f& b, const Eigen::Isometry3f& cam_T_model, const Intrinsics& k,
             const Rgb& colour)
{
    static const int edges[12][2] = { { 0, 1 }, { 1, 3 }, { 3, 2 }, { 2, 0 }, { 4, 5 }, { 5, 7 },
                                      { 7, 6 }, { 6, 4 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } };
    float u[8], v[8];
    bool  ok[8];
    for (int i = 0; i < 8; ++i)
    {
        ok[i] = project(cam_T_model * b.corner(static_cast<Eigen::AlignedBox3f::CornerType>(i)), k, u[i], v[i]);
    }
    for (const auto& e : edges)
    {
        if (ok[e[0]] && ok[e[1]])
        {
            c.line(u[e[0]], v[e[0]], u[e[1]], v[e[1]], colour);
        }
    }
}

}  // namespace

std::array<std::uint8_t, 3> debugColour(std::size_t index)
{
    static const Rgb palette[] = { { 0, 200, 255 }, { 255, 190, 0 }, { 90, 255, 120 }, { 190, 120, 255 } };
    return palette[index % 4];
}

void drawText(std::vector<std::uint8_t>& rgb, int width, int height, int x, int y, const std::string& text,
              const std::array<std::uint8_t, 3>& colour)
{
    Canvas c{ rgb, width, height };
    c.darken(x - 2, y - 1, x + kGlyphW * static_cast<int>(text.size()) + 2, y + kGlyphH + 1);
    for (std::size_t i = 0; i < text.size(); ++i)
    {
        const int ch = static_cast<unsigned char>(text[i]);
        if (ch < 32 || ch > 126)
        {
            continue;
        }
        for (int row = 0; row < kGlyphH; ++row)
        {
            for (int col = 0; col < kGlyphW; ++col)
            {
                if (kGlyphs[ch - 32][row] & (1u << col))
                {
                    c.set(x + static_cast<int>(i) * kGlyphW + col, y + row, colour);
                }
            }
        }
    }
}

void renderDebugImage(const DepthImage& depth, const Intrinsics& k_depth, const std::vector<DebugObject>& objects,
                      float free_space_tol, std::vector<std::uint8_t>& rgb, int& width, int& height)
{
    // Drawn at least 640 wide: the tracker reads a half-resolution image, too small for text.
    const int  s = std::max(1, (639 + depth.width) / depth.width);
    Intrinsics k = k_depth;
    k.fx *= s, k.fy *= s, k.cx = (k.cx + 0.5f) * s - 0.5f, k.cy = (k.cy + 0.5f) * s - 0.5f;
    k.width *= s, k.height *= s;
    const int w = k.width, h = k.height;
    width = w, height = h;
    const auto zat = [&](int u, int v) { return depth.at(u / s, v / s); };
    rgb.assign(static_cast<std::size_t>(w) * h * 3, 0);
    Canvas c{ rgb, w, h };

    // Depth in grey, near bright, over the range the image actually spans.
    float zlo = 1e9f, zhi = 0.0f;
    for (int v = 0; v < h; v += 4)
    {
        for (int u = 0; u < w; u += 4)
        {
            const float z = zat(u, v);
            if (z > 0.0f && std::isfinite(z))
            {
                zlo = std::min(zlo, z), zhi = std::max(zhi, z);
            }
        }
    }
    zhi = std::min(zhi, zlo + 3.0f);
    const float span = std::max(0.05f, zhi - zlo);
    for (int v = 0; v < h; ++v)
    {
        for (int u = 0; u < w; ++u)
        {
            const float z = zat(u, v);
            if (!(z > 0.0f) || !std::isfinite(z))
            {
                c.set(u, v, { 20, 20, 45 });  // no reading
                continue;
            }
            const float t = std::clamp((z - zlo) / span, 0.0f, 1.0f);
            const auto  g = static_cast<std::uint8_t>(215 - 175 * t);
            c.set(u, v, { g, g, g });
        }
    }

    for (std::size_t i = 0; i < objects.size(); ++i)
    {
        const DebugObject& o      = objects[i];
        const Rgb          colour = debugColour(i);
        if (o.model == nullptr || o.out == nullptr)
        {
            continue;
        }
        const ObjectTracker::Output& out = *o.out;
        // Searching: the poses being tried, dim.
        const Rgb dim{ static_cast<std::uint8_t>(colour[0] / 2), static_cast<std::uint8_t>(colour[1] / 2),
                       static_cast<std::uint8_t>(colour[2] / 2) };
        for (const auto& hyp : o.hypotheses)
        {
            drawBox(c, o.model->bounds(), hyp, k, dim);
        }
        if (out.state != ObjectTracker::State::kTracking && !o.valid)
        {
            continue;
        }
        // The fitted points inside the part's box that did not land on it.
        for (std::size_t p = 0; p < out.debug_points.size(); ++p)
        {
            float u, v;
            if (out.debug_inlier[p] == kDebugOffModel && project(out.debug_points[p], k, u, v))
            {
                c.set(static_cast<int>(u), static_cast<int>(v), kMagenta);
            }
        }
        // The model's camera-facing surface at the fitted pose, judged against the depth.
        const auto&           vp   = o.model->verifyPoints();
        const auto&           vn   = o.model->verifyNormals();
        const Eigen::Matrix3f R    = out.cam_T_model.linear();
        const int             size = std::max(1, w / 400);
        for (std::size_t p = 0; p < vp.size(); ++p)
        {
            const Eigen::Vector3f q = out.cam_T_model * vp[p];
            if (o.model->normalsOriented() && (R * vn[p]).dot(q) >= 0.0f)
            {
                continue;
            }
            float u, v;
            if (!project(q, k, u, v) || u < 0 || v < 0 || u >= w || v >= h)
            {
                continue;
            }
            const float z = zat(static_cast<int>(u), static_cast<int>(v));
            if (!(z > 0.0f) || !std::isfinite(z) || z < q.z() - free_space_tol)
            {
                continue;  // no reading, or something in front: says nothing
            }
            c.dot(static_cast<int>(u), static_cast<int>(v), std::abs(z - q.z()) <= free_space_tol ? colour : kRed,
                  size);
        }
        if (!out.roi.empty())
        {
            c.rect(out.roi.u0 * s, out.roi.v0 * s, out.roi.u1 * s - 1, out.roi.v1 * s - 1, colour, o.valid ? 0 : 4);
            drawText(rgb, w, h, out.roi.u0 * s + 3, std::max(0, out.roi.v0 * s - 14), o.name, colour);
        }
    }

    // Status, one line per object.
    int y = 4;
    for (std::size_t i = 0; i < objects.size(); ++i)
    {
        const DebugObject& o = objects[i];
        if (o.out == nullptr)
        {
            continue;
        }
        const ObjectTracker::Output& out = *o.out;
        char                         line[160];
        if (out.state == ObjectTracker::State::kTracking)
        {
            std::snprintf(line, sizeof line, "%-7s %s ratio %.2f cov %.2f viol %.2f %4dpts %4.1fms", o.name.c_str(),
                          o.valid ? "TRACK " : "REJECT", out.score.ratio, out.score.coverage, out.score.violation,
                          out.scene_points, out.ms);
        }
        else
        {
            std::snprintf(line, sizeof line, "%-7s SEARCH  %zu poses tried", o.name.c_str(), o.hypotheses.size());
        }
        drawText(rgb, w, h, 6, y, line, o.valid ? debugColour(i) : (out.state == ObjectTracker::State::kTracking ? kRed : kWhite));
        y += 15;
    }
    drawText(rgb, w, h, 6, h - 16, "model: colour=confirmed RED=seen-through  MAGENTA: points off model", kWhite);
}

}  // namespace g1_perception
