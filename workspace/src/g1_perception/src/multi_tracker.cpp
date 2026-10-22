#include "g1_perception/multi_tracker.hpp"

#include <map>
#include <set>
#include <stdexcept>

namespace g1_perception
{

std::vector<ObjectConfig> expandInstances(const ObjectConfig& c, int n)
{
    if (n <= 1)
    {
        return { c };
    }
    std::vector<ObjectConfig> out;
    for (int i = 1; i <= n; ++i)
    {
        ObjectConfig copy = c;
        copy.name         = c.name + "_" + std::to_string(i);
        copy.group        = c.group.empty() ? c.name : c.group;
        out.push_back(std::move(copy));
    }
    return out;
}

MultiTracker::MultiTracker(std::vector<ObjectConfig> objects)
{
    std::set<std::string> names;
    for (const auto& c : objects)
    {
        if (c.name.empty() || !names.insert(c.name).second)
        {
            throw std::runtime_error("object names must be non-empty and unique ('" + c.name + "')");
        }
    }
    std::map<std::string, std::shared_ptr<const ObjectModel>> models;
    for (auto& c : objects)
    {
        for (const auto& p : c.priors)
        {
            if (p.frame == c.name)
            {
                throw std::runtime_error("object '" + c.name + "' has a prior relative to itself");
            }
        }
        Object o;
        // Instances of one part load it once.
        const std::string key = c.model_path + "|" + std::to_string(c.model.scale) + "|" +
                                std::to_string(c.model.voxel) + "|" + std::to_string(c.model.grid_cell);
        std::shared_ptr<const ObjectModel> model;
        try
        {
            auto& cached = models[key];
            if (!cached)
            {
                cached = ObjectModel::load(c.model_path, c.model);
            }
            model = cached;
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error("object '" + c.name + "': " + e.what());
        }
        // The extent in the output frame: the model's points carried into it, boxed there.
        Eigen::AlignedBox3f     box;
        const Eigen::Isometry3f output_T_model = c.model_T_output.inverse();
        for (const auto& p : model->points())
        {
            box.extend(output_T_model * p);
        }
        o.box_centre = box.center();
        o.box_size   = box.sizes();
        o.tracker    = std::make_unique<ObjectTracker>(c.name, model, c.params);
        o.group      = c.group.empty() ? c.name : c.group;
        o.config     = std::move(c);
        objects_.push_back(std::move(o));
    }
}

void MultiTracker::reset()
{
    for (auto& o : objects_)
    {
        o.tracker->reset();
        o.motion.reset();
        o.world_T_model.reset();
        o.was_tracking = false;
    }
}

void MultiTracker::setDebug(bool on)
{
    debug_ = on;
    for (auto& o : objects_)
    {
        o.tracker->params().keep_debug = on;
    }
}

std::vector<MultiTracker::Result> MultiTracker::update(const DepthImage& depth, const Intrinsics& k, double stamp,
                                                       const Eigen::Isometry3f& world_T_cam,
                                                       const std::string& camera_frame, const FrameLookup& lookup)
{
    const Eigen::Isometry3f cam_T_world = world_T_cam.inverse();
    // One lookup per frame name per frame: several priors usually share a frame.
    std::map<std::string, std::optional<Eigen::Isometry3f>> frame_cache;
    const auto world_T = [&](const std::string& frame) -> std::optional<Eigen::Isometry3f> {
        for (const auto& o : objects_)
        {
            if (o.config.name == frame)
            {
                return o.world_T_model;
            }
        }
        auto it = frame_cache.find(frame);
        if (it == frame_cache.end())
        {
            it = frame_cache.emplace(frame, lookup ? lookup(frame, stamp) : std::nullopt).first;
        }
        return it->second;
    };

    // Another instance of the same part already at (within `sep` of) this world position.
    const auto held_by_sibling = [&](const Object& self, const Eigen::Vector3f& world_p) {
        for (const auto& other : objects_)
        {
            if (&other != &self && other.group == self.group && other.world_T_model &&
                (other.world_T_model->translation() - world_p).norm() < self.config.instance_separation)
            {
                return true;
            }
        }
        return false;
    };

    std::vector<Result> results;
    results.reserve(objects_.size());
    for (auto& o : objects_)
    {
        FrameContext ctx;
        ctx.stamp = stamp;
        if (const auto p = o.motion.predict(stamp))
        {
            ctx.predicted = cam_T_world * *p;
        }
        if (o.world_T_model)
        {
            ctx.fallbacks.push_back(cam_T_world * *o.world_T_model);  // as last seen, no velocity
        }
        ctx.motion = [&](double then) -> std::optional<Eigen::Isometry3f> {
            if (!lookup)
            {
                return std::nullopt;
            }
            const auto world_T_cam_then = lookup(camera_frame, then);
            if (!world_T_cam_then)
            {
                return std::nullopt;
            }
            return cam_T_world * *world_T_cam_then;  // cam_now_T_cam_then
        };
        for (const auto& other : objects_)
        {
            if (&other != &o && other.world_T_model)
            {
                ctx.others.push_back({ &other.tracker->model(), cam_T_world * *other.world_T_model });
            }
        }
        if (o.tracker->state() == ObjectTracker::State::kSearching)
        {
            for (const auto& p : o.config.priors)
            {
                if (const auto parent = world_T(p.frame))
                {
                    const Eigen::Isometry3f world_T_prior = *parent * p.pose;
                    if (!held_by_sibling(o, world_T_prior.translation()))
                    {
                        ctx.priors.push_back(cam_T_world * world_T_prior);
                    }
                }
            }
        }

        Result r;
        r.name         = o.config.name;
        r.out          = o.tracker->update(depth, k, ctx);
        if (r.out.state == ObjectTracker::State::kTracking && r.out.valid)
        {
            // A fit the depth confirms can still be the wrong thing: another instance's part, or
            // a look-alike where this object cannot be. Either way it is dropped and searched
            // for again, rather than reported.
            const Eigen::Isometry3f world_T_fit = world_T_cam * r.out.cam_T_model;
            bool                    ok          = !held_by_sibling(o, world_T_fit.translation());
            if (ok && o.config.region)
            {
                const auto parent = world_T(o.config.region->frame);
                ok = parent && o.config.region->contains(parent->inverse() * world_T_fit * o.config.model_T_output);
            }
            if (!ok)
            {
                o.tracker->reset();
                r.out.valid = false;
                r.out.state = ObjectTracker::State::kSearching;
                r.rejected  = true;
            }
        }
        const bool tracking = r.out.state == ObjectTracker::State::kTracking;
        r.valid        = tracking && r.out.valid;
        r.box_centre   = o.box_centre;
        r.box_size     = o.box_size;
        if (debug_ && r.out.state == ObjectTracker::State::kSearching)
        {
            for (const auto& prior : ctx.priors)
            {
                for (const auto& off : o.tracker->params().prior_offsets)
                {
                    r.hypotheses.push_back(prior * off);
                }
            }
        }
        if (r.valid)
        {
            o.world_T_model = world_T_cam * r.out.cam_T_model;
            o.motion.update(stamp, *o.world_T_model);
            r.world_T_output = *o.world_T_model * o.config.model_T_output;
            r.cam_T_output   = r.out.cam_T_model * o.config.model_T_output;
        }
        else if (!tracking)
        {
            // Lost: stop explaining depth with a pose nobody has confirmed, and stop predicting.
            o.world_T_model.reset();
            o.motion.reset();
        }
        r.just_locked  = tracking && !o.was_tracking;
        r.just_lost    = !tracking && o.was_tracking;
        o.was_tracking = tracking;
        results.push_back(std::move(r));
    }
    return results;
}

}  // namespace g1_perception
