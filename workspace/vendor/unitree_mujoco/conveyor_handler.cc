#include "conveyor_handler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace grove_g1
{
namespace
{

// Belt geoms this manager owns are named with this prefix in the scene (an exact match, or
// `conveyor_belt_<anything>` for more than one belt). Anything else in the model is left
// alone.
constexpr const char* kBeltPrefix = "conveyor_belt";

constexpr int kTickRateHz = 500;  // matches dex3_handler's own rate; this is a force, not a DDS write

struct Config
{
    bool enabled = false;

    // Target speed along the belt's local +x, m/s. Positive only: a belt that should run the
    // other way is turned around in the scene, not given a negative speed here, so the geom's
    // own pose stays the single source of truth for direction.
    double speed_mps = 0.08;

    // N per (m/s) of velocity error on the driven body's linear dofs. Large enough to reach
    // speed_mps well inside one capture pass at a typical small-part mass (tens of grams to a
    // few hundred), not tuned against any real belt's drive torque.
    double gain = 40.0;

    // How far above the belt's own top surface a body's origin may sit and still count as
    // "on" it, in metres. Covers a settling bounce without reaching so high that a part lifted
    // off by the arm is still dragged.
    double capture_height_m = 0.09;

    // Slotted pallet (see the Pallet struct below). Only used when the scene declares one.
    bool   pallet_enabled  = true;
    double pallet_speed_mps = 0.08;  // peak speed of an index move
    double pallet_settle_s  = 1.0;   // a panel must sit in the station nest this long, untouched
    double pallet_kp        = 4000.0;  // N/m on the slide joint; ~20 rad/s with the 10 kg pallet
    double pallet_kd        = 400.0;   // N/(m/s); critically damped with kp at 10 kg
    // A panel counts as "at rest" below these, relative to the stationary pallet.
    double pallet_rest_lin_mps   = 0.02;
    double pallet_rest_ang_radps = 0.3;
};

// One belt zone: a static geom's body, half-extents, and the direction to drive toward.
struct Belt
{
    int    body_id = -1;
    double half_x  = 0.0;
    double half_y  = 0.0;
    double half_z  = 0.0;
    std::string name;
};

// One body this manager may drive: resolved once from the model's free joints, independent of
// whether it is currently on a belt -- that is re-checked every tick, since parts move on and
// off a belt as the scene runs.
struct Draggable
{
    int body_id = -1;
    int dof_adr = -1;  // first of 6 free-joint dofs; [0..2] are world-frame linear velocity
};

// The slotted pallet (tools/make_conveyor_pallet.py in g1_bringup writes it into the scene).
// It rides a slide joint, never a free joint, so it cannot yaw, tip or drift: its pose is always
// exactly known, which is the point of a fixture. It is driven here by a PD force on that joint
// following a min-jerk profile, and indexed one nest at a time: when a panel has sat in the
// nest at the load station, at rest and untouched by the robot, for settle_s, the next empty
// nest is brought to the station. After the last nest it runs to its joint's upper limit (the
// belt's far end) and stays there.
struct Pallet
{
    bool   present = false;
    int    joint_id = -1;
    int    qpos_adr = -1;
    int    dof_adr  = -1;
    int    body_id  = -1;
    double q_max    = 0.0;
    double half_x   = 0.0;  // plate half-extents, for keeping the belt's drag off its cargo
    double half_y   = 0.0;
    std::vector<int> nest_sites;  // [0] = conveyor_nest_1, the downstream end
    int    station_site = -1;
    int    robot_root   = -1;  // root body of the robot, to tell "placed" from "still held"

    // Runtime state, all in simulation time.
    std::size_t station_nest = 0;  // index into nest_sites currently at the station
    bool   finished  = false;
    bool   moving    = false;
    double hold_q    = 0.0;
    double q0 = 0.0, q1 = 0.0, t0 = 0.0, duration = 0.0;
    double settled_since = -1.0;
    double last_time     = -1.0;
};

struct State
{
    std::thread       thread;
    std::atomic<bool> running{ false };
};

State& state()
{
    // Deliberately leaked, for the same reason the other handlers leak theirs: the physics
    // thread ends in exit(0), which runs static destructors while this thread may still be
    // alive, and destroying a joinable std::thread calls std::terminate.
    static State* s = new State();
    return *s;
}

Config loadConfig()
{
    Config      cfg;
    const char* path = std::getenv("GROVE_G1_SENSOR_CONFIG");
    if (path == nullptr || *path == '\0')
    {
        return cfg;
    }
    try
    {
        const YAML::Node root = YAML::LoadFile(path);
        const YAML::Node node = root["conveyor"];
        if (!node)
        {
            return cfg;
        }
        cfg.enabled = node["enabled"] ? node["enabled"].as<bool>() : true;
        if (node["speed_mps"])
        {
            cfg.speed_mps = node["speed_mps"].as<double>();
        }
        if (node["gain"])
        {
            cfg.gain = node["gain"].as<double>();
        }
        if (node["capture_height_m"])
        {
            cfg.capture_height_m = node["capture_height_m"].as<double>();
        }
        if (const YAML::Node pallet = node["pallet"])
        {
            cfg.pallet_enabled = pallet["enabled"] ? pallet["enabled"].as<bool>() : true;
            if (pallet["speed_mps"]) cfg.pallet_speed_mps = pallet["speed_mps"].as<double>();
            if (pallet["settle_s"]) cfg.pallet_settle_s = pallet["settle_s"].as<double>();
            if (pallet["kp"]) cfg.pallet_kp = pallet["kp"].as<double>();
            if (pallet["kd"]) cfg.pallet_kd = pallet["kd"].as<double>();
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(
            stderr, "[grove_g1] conveyor config '%s' failed to load (%s); DISABLED\n", path,
            e.what());
        cfg.enabled = false;
    }
    if (cfg.pallet_speed_mps <= 0.0 || cfg.pallet_settle_s < 0.0 || cfg.pallet_kp <= 0.0 ||
        cfg.pallet_kd < 0.0)
    {
        std::fprintf(
            stderr,
            "[grove_g1] conveyor pallet needs speed_mps > 0, settle_s >= 0, kp > 0, kd >= 0; "
            "pallet DISABLED\n");
        cfg.pallet_enabled = false;
    }
    if (cfg.speed_mps <= 0.0 || cfg.gain <= 0.0 || cfg.capture_height_m <= 0.0)
    {
        std::fprintf(
            stderr,
            "[grove_g1] conveyor needs speed_mps, gain and capture_height_m all > 0; DISABLED\n");
        cfg.enabled = false;
    }
    return cfg;
}

bool resolve(const mjModel* model, std::vector<Belt>& belts, std::vector<Draggable>& bodies)
{
    belts.clear();
    bodies.clear();
    const std::size_t prefix_len = std::char_traits<char>::length(kBeltPrefix);

    for (int g = 0; g < model->ngeom; ++g)
    {
        const char* name = mj_id2name(model, mjOBJ_GEOM, g);
        if (name == nullptr || std::string(name).compare(0, prefix_len, kBeltPrefix) != 0)
        {
            continue;
        }
        if (model->geom_type[g] != mjGEOM_BOX)
        {
            std::fprintf(stderr, "[grove_g1] conveyor geom '%s' is not a box; not managed\n", name);
            continue;
        }
        Belt belt;
        belt.body_id = model->geom_bodyid[g];
        belt.half_x  = model->geom_size[3 * g + 0];
        belt.half_y  = model->geom_size[3 * g + 1];
        belt.half_z  = model->geom_size[3 * g + 2];
        belt.name    = name;
        belts.push_back(belt);
    }
    if (belts.empty())
    {
        return false;
    }

    for (int j = 0; j < model->njnt; ++j)
    {
        if (model->jnt_type[j] != mjJNT_FREE)
        {
            continue;
        }
        Draggable body;
        body.body_id = model->jnt_bodyid[j];
        body.dof_adr = model->jnt_dofadr[j];
        bodies.push_back(body);
    }

    std::fprintf(
        stderr, "[grove_g1] conveyor: %zu belt(s), %zu free body(ies) eligible to drag\n",
        belts.size(), bodies.size());
    return !bodies.empty();
}

// Finds the pallet the generator writes. Absent is normal (every scene but the conveyor one);
// present but malformed is reported and ignored, never half-driven.
void resolvePallet(const mjModel* model, Pallet& p)
{
    p = Pallet{};
    const int j = mj_name2id(model, mjOBJ_JOINT, "conveyor_pallet_slide");
    if (j < 0)
    {
        return;
    }
    const int plate = mj_name2id(model, mjOBJ_GEOM, "conveyor_pallet_plate");
    const int station = mj_name2id(model, mjOBJ_SITE, "conveyor_load_station");
    if (model->jnt_type[j] != mjJNT_SLIDE || !model->jnt_limited[j] || plate < 0 || station < 0)
    {
        std::fprintf(
            stderr,
            "[grove_g1] conveyor_pallet_slide found but the pallet is malformed (needs a limited "
            "slide joint, conveyor_pallet_plate and conveyor_load_station); pallet is OFF\n");
        return;
    }
    for (int k = 1;; ++k)
    {
        const std::string name = "conveyor_nest_" + std::to_string(k);
        const int         id   = mj_name2id(model, mjOBJ_SITE, name.c_str());
        if (id < 0)
        {
            break;
        }
        p.nest_sites.push_back(id);
    }
    if (p.nest_sites.empty())
    {
        std::fprintf(stderr, "[grove_g1] pallet has no conveyor_nest_<k> sites; pallet is OFF\n");
        return;
    }
    p.joint_id     = j;
    p.qpos_adr     = model->jnt_qposadr[j];
    p.dof_adr      = model->jnt_dofadr[j];
    p.body_id      = model->jnt_bodyid[j];
    p.q_max        = model->jnt_range[2 * j + 1];
    p.half_x       = model->geom_size[3 * plate + 0];
    p.half_y       = model->geom_size[3 * plate + 1];
    p.station_site = station;
    const int pelvis = mj_name2id(model, mjOBJ_BODY, "pelvis");
    p.robot_root     = pelvis >= 0 ? model->body_rootid[pelvis] : -1;
    p.present        = true;
    std::fprintf(
        stderr, "[grove_g1] conveyor pallet: %zu nests, travel 0..%.3f m\n", p.nest_sites.size(),
        p.q_max);
}

// True if `world_pos` lies inside site `site`'s box (sites are generated as boxes).
bool insideSite(const mjModel* m, const mjData* d, int site, const double world_pos[3])
{
    double delta[3];
    mju_sub3(delta, world_pos, d->site_xpos + 3 * site);
    const mjtNum* R = d->site_xmat + 9 * site;
    for (int i = 0; i < 3; ++i)
    {
        const double local = R[i] * delta[0] + R[3 + i] * delta[1] + R[6 + i] * delta[2];
        if (std::abs(local) > m->site_size[3 * site + i])
        {
            return false;
        }
    }
    return true;
}

// Is a free body sitting in the station nest, at rest, with no robot geom touching it?
bool stationNestFilled(
    const mjModel* m, const mjData* d, const Pallet& p, const std::vector<Draggable>& bodies,
    const Config& cfg)
{
    const int site = p.nest_sites[p.station_nest];
    for (const Draggable& body : bodies)
    {
        if (!insideSite(m, d, site, d->xpos + 3 * body.body_id))
        {
            continue;
        }
        const double* v = d->qvel + body.dof_adr;
        if (mju_norm3(v) > cfg.pallet_rest_lin_mps || mju_norm3(v + 3) > cfg.pallet_rest_ang_radps)
        {
            continue;
        }
        bool held = false;
        for (int c = 0; c < d->ncon && p.robot_root >= 0 && !held; ++c)
        {
            const int b1 = m->geom_bodyid[d->contact[c].geom[0]];
            const int b2 = m->geom_bodyid[d->contact[c].geom[1]];
            held = (b1 == body.body_id && m->body_rootid[b2] == p.robot_root) ||
                   (b2 == body.body_id && m->body_rootid[b1] == p.robot_root);
        }
        if (!held)
        {
            return true;
        }
    }
    return false;
}

// One pallet tick: advance the state machine, then write the drive force.
void tickPallet(const mjModel* m, mjData* d, Pallet& p, const std::vector<Draggable>& bodies,
                const Config& cfg)
{
    const double t = d->time;
    if (t + 1e-9 < p.last_time)
    {
        // Simulation reset (Backspace in the viewer): back to nest 1 at the station.
        p.station_nest = 0;
        p.finished     = false;
        p.moving       = false;
        p.hold_q       = d->qpos[p.qpos_adr];
        p.settled_since = -1.0;
        std::fprintf(stderr, "[grove_g1] pallet: simulation reset, nest 1 at the load station\n");
    }
    if (p.last_time < 0.0)
    {
        p.hold_q = d->qpos[p.qpos_adr];
    }
    p.last_time = t;

    if (!p.moving && !p.finished)
    {
        if (stationNestFilled(m, d, p, bodies, cfg))
        {
            if (p.settled_since < 0.0)
            {
                p.settled_since = t;
            }
            else if (t - p.settled_since >= cfg.pallet_settle_s)
            {
                const double* axis = d->xaxis + 3 * p.joint_id;  // slide direction, world frame
                p.q0 = p.hold_q;
                if (p.station_nest + 1 < p.nest_sites.size())
                {
                    const int next = p.nest_sites[p.station_nest + 1];
                    double    gap[3];
                    mju_sub3(gap, d->site_xpos + 3 * p.station_site, d->site_xpos + 3 * next);
                    p.q1 = std::min(p.q_max, p.hold_q + mju_dot3(gap, axis));
                    ++p.station_nest;
                    std::fprintf(
                        stderr, "[grove_g1] pallet: nest %zu filled, bringing nest %zu to the station\n",
                        p.station_nest, p.station_nest + 1);
                }
                else
                {
                    p.q1       = p.q_max;
                    p.finished = true;
                    std::fprintf(
                        stderr, "[grove_g1] pallet: all %zu nests filled, running to the end\n",
                        p.nest_sites.size());
                }
                // Min-jerk peaks at 1.875x the mean speed; size the move so the peak is speed_mps.
                p.duration = std::max(0.5, 1.875 * std::abs(p.q1 - p.q0) / cfg.pallet_speed_mps);
                p.t0       = t;
                p.moving   = true;
                p.settled_since = -1.0;
            }
        }
        else
        {
            p.settled_since = -1.0;
        }
    }

    double q_ref = p.hold_q, v_ref = 0.0, a_ref = 0.0;
    if (p.moving)
    {
        const double s  = std::clamp((t - p.t0) / p.duration, 0.0, 1.0);
        const double dq = p.q1 - p.q0;
        const double T  = p.duration;
        q_ref = p.q0 + dq * (10 * s * s * s - 15 * s * s * s * s + 6 * s * s * s * s * s);
        v_ref = dq * (30 * s * s - 60 * s * s * s + 30 * s * s * s * s) / T;
        a_ref = dq * (60 * s - 180 * s * s + 120 * s * s * s) / (T * T);
        if (s >= 1.0)
        {
            p.moving = false;
            p.hold_q = p.q1;
            v_ref = a_ref = 0.0;
        }
    }
    const double q = d->qpos[p.qpos_adr];
    const double v = d->qvel[p.dof_adr];
    d->qfrc_applied[p.dof_adr] = cfg.pallet_kp * (q_ref - q) + cfg.pallet_kd * (v_ref - v) +
                                 m->body_mass[p.body_id] * a_ref;
}

// True if `world_pos` is above the pallet's plate: the pallet carries whatever is there, so the
// belt's own drag must leave it alone -- including a panel still in the robot's hand on its way
// into a nest, which the drag would otherwise push off target.
bool overPallet(const mjData* d, const Pallet& p, const double world_pos[3])
{
    if (!p.present)
    {
        return false;
    }
    double delta[3];
    mju_sub3(delta, world_pos, d->xpos + 3 * p.body_id);
    const mjtNum* R = d->xmat + 9 * p.body_id;
    const double  x = R[0] * delta[0] + R[3] * delta[1] + R[6] * delta[2];
    const double  y = R[1] * delta[0] + R[4] * delta[1] + R[7] * delta[2];
    const double  z = R[2] * delta[0] + R[5] * delta[1] + R[8] * delta[2];
    return std::abs(x) <= p.half_x && std::abs(y) <= p.half_y && z >= 0.0;
}

// `world_pos` expressed in `belt`'s own body frame, using that body's current world pose --
// so a belt the scene rotates or (one day) moves still drags along its own local +x, not a
// world axis baked in at compile time.
void toBeltFrame(const mjData* data, const Belt& belt, const double world_pos[3], double out[3])
{
    double delta[3];
    mju_sub3(delta, world_pos, data->xpos + 3 * belt.body_id);
    const mjtNum* R = data->xmat + 9 * belt.body_id;  // row-major 3x3, world <- local
    // out = R^T * delta
    out[0] = R[0] * delta[0] + R[3] * delta[1] + R[6] * delta[2];
    out[1] = R[1] * delta[0] + R[4] * delta[1] + R[7] * delta[2];
    out[2] = R[2] * delta[0] + R[5] * delta[1] + R[8] * delta[2];
}

void run(const Config cfg, mjModel** model, mjData** data, std::recursive_mutex* sim_mtx)
{
    auto& s = state();
    while (s.running.load(std::memory_order_relaxed) && (*model == nullptr || *data == nullptr))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!s.running.load(std::memory_order_relaxed))
    {
        return;
    }

    mjModel*               m = *model;
    std::vector<Belt>      belts;
    std::vector<Draggable> bodies;
    if (!resolve(m, belts, bodies))
    {
        std::fprintf(stderr, "[grove_g1] no conveyor belts (or no free bodies) in this scene; conveyor is OFF\n");
        return;
    }
    Pallet pallet;
    if (cfg.pallet_enabled)
    {
        resolvePallet(m, pallet);
    }
    const auto period = std::chrono::nanoseconds(std::chrono::seconds(1)) / kTickRateHz;
    auto       next    = std::chrono::steady_clock::now();

    while (s.running.load(std::memory_order_relaxed))
    {
        next += period;
        std::this_thread::sleep_until(next);

        std::lock_guard<std::recursive_mutex> lock(*sim_mtx);
        if (*model != m || *data == nullptr)
        {
            std::fprintf(stderr, "[grove_g1] model replaced; conveyor is OFF\n");
            return;
        }
        mjData* d = *data;

        if (pallet.present)
        {
            tickPallet(m, d, pallet, bodies, cfg);
        }

        for (const Draggable& body : bodies)
        {
            const double* pos = d->xpos + 3 * body.body_id;
            bool          on_belt = false;
            if (overPallet(d, pallet, pos))
            {
                mju_zero3(d->qfrc_applied + body.dof_adr);
                continue;
            }
            double        dir_world[3] = { 0.0, 0.0, 0.0 };

            for (const Belt& belt : belts)
            {
                double local[3];
                toBeltFrame(d, belt, pos, local);
                const double height_above_top = local[2] - belt.half_z;
                if (std::abs(local[0]) <= belt.half_x && std::abs(local[1]) <= belt.half_y &&
                    height_above_top >= 0.0 && height_above_top <= cfg.capture_height_m)
                {
                    const mjtNum* R = d->xmat + 9 * belt.body_id;
                    dir_world[0]    = R[0];  // belt body's local +x, in world frame
                    dir_world[1]    = R[3];
                    dir_world[2]    = R[6];
                    on_belt         = true;
                    break;  // first matching belt wins; scenes should not overlap belts
                }
            }

            double* force = d->qfrc_applied + body.dof_adr;
            if (!on_belt)
            {
                mju_zero3(force);
                continue;
            }
            const double* vel = d->qvel + body.dof_adr;
            for (int k = 0; k < 3; ++k)
            {
                const double target = dir_world[k] * cfg.speed_mps;
                force[k]            = cfg.gain * (target - vel[k]);
            }
        }
    }

    std::lock_guard<std::recursive_mutex> lock(*sim_mtx);
    if (*model == m && *data != nullptr)
    {
        for (const Draggable& body : bodies)
        {
            mju_zero3((*data)->qfrc_applied + body.dof_adr);
        }
        if (pallet.present)
        {
            (*data)->qfrc_applied[pallet.dof_adr] = 0.0;
        }
    }
}

}  // namespace

void StartConveyorHandler(mjModel** model, mjData** data, std::recursive_mutex* sim_mtx)
{
    auto& s = state();
    if (s.running.load(std::memory_order_relaxed))
    {
        return;
    }
    const Config cfg = loadConfig();
    if (!cfg.enabled)
    {
        return;
    }
    s.running.store(true, std::memory_order_relaxed);
    s.thread = std::thread(run, cfg, model, data, sim_mtx);
}

void StopConveyorHandler()
{
    auto& s = state();
    if (!s.thread.joinable())
    {
        s.running.store(false, std::memory_order_relaxed);
        return;
    }
    s.running.store(false, std::memory_order_relaxed);
    s.thread.join();
}

}  // namespace grove_g1
