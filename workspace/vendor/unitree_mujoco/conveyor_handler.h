#ifndef GROVE_G1_CONVEYOR_HANDLER_H_
#define GROVE_G1_CONVEYOR_HANDLER_H_

// SIMULATION ONLY. Drags any freely-jointed body resting on a named belt geom along that
// geom's local +x axis at a configured speed, standing in for a real conveyor's friction
// drive the way grasp_weld stands in for finger contact.
//
// MuJoCo has no native "infinite moving surface": a geom's own position is the belt's only
// physical truth, and nothing short of actually sliding it (which a finite-range slide joint
// cannot do forever) moves a textured strip across it. This instead treats the belt as a zone:
// each tick, any free body whose origin falls inside a geom named `conveyor_belt` (or
// `conveyor_belt_*`, so a scene may declare more than one) gets a world-frame force on its
// linear dofs driving its velocity toward the belt's own direction and speed, exactly as
// grasp_weld drives a weld rather than scripting a trajectory. A body the robot is holding (or
// anything without a free joint) is untouched.
//
// Direction and the belt's own motion are both taken from the geom's pose, not hardcoded: a
// belt rotated in the scene drags along its rotated local +x, so moving or re-orienting the
// belt in the MJCF never needs a matching code change.
//
// It also drives a slotted pallet when the scene has one (g1_bringup's
// tools/make_conveyor_pallet.py writes it): a fixture on a slide joint, indexed one nest at a
// time to a fixed load station as panels are placed, then run to the belt's end. Bodies above
// the pallet are excluded from the belt drag, since the pallet carries them. See
// conveyor_handler.cc's Pallet struct.
//
// On hardware this file never runs. The belt's own motor and friction do this job, and
// speed/gain here are properties of the simulated zone, not of any real conveyor -- same
// hardware re-validation list as grasp_weld's thresholds.
//
// Lives outside the vendored sources so the patch against them stays a few lines; see
// workspace/patches/unitree_mujoco/README.md.

#include <mujoco/mujoco.h>

#include <mutex>

namespace grove_g1
{

// Starts the conveyor thread if the config enables it and the scene declares at least one
// `conveyor_belt*` geom, otherwise returns having done nothing. Non-blocking.
//
// model/data are the addresses of unitree_mujoco's globals, not their values: a model dropped
// into the viewer reassigns both. sim_mtx is unitree_mujoco's own simulation mutex, held for
// the whole tick for the same reason grasp_weld holds it: this writes qfrc_applied for bodies
// physics is about to step, and a snapshot-then-write-back could race that step.
void StartConveyorHandler(mjModel** model, mjData** data, std::recursive_mutex* sim_mtx);

// Stops the conveyor thread and BLOCKS until it has terminated. Clears qfrc_applied on every
// body it was driving on the way out, so a stopped manager does not leave a part accelerating
// forever. Safe to call when nothing is running.
//
// Must be called before mj_deleteModel on any path that replaces the model: body and dof ids
// are resolved once and cached, so they outlive the model they came from.
void StopConveyorHandler();

}  // namespace grove_g1

#endif  // GROVE_G1_CONVEYOR_HANDLER_H_
