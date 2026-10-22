# g1_hand_interface

`ros2_control` `SystemInterface` for one Inspire RH56E2 hand, over a `HandCmd` / `HandState` topic
pair.

`ament_cmake`, C++20. The plugin is `g1_hand_interface/G1HandSystem`. It used to be
`G1Dex3System`, for Unitree's Dex3-1; the wire handling is the same and the joint set is not.

## What it talks to, and what that is not

The RH56E2 is not a Unitree device and does not speak DDS. This component speaks Unitree's
`HandCmd` and `HandState` message types (the packed mode byte, the positional `motor_cmd` array)
on `/rh56e2/<side>/{cmd,state}`, and something else answers:

| Where | Who answers |
|---|---|
| Simulation | `unitree_mujoco`, through `workspace/vendor/unitree_mujoco/dex3_handler.cc` and its patches. |
| The robot | **Nothing yet.** A driver node has to serve the same two topics and translate to the hand's own interface. It is the next piece of work, and until it exists this component cannot activate on hardware: it refuses to without fresh `HandState`. |

Keeping the contract generic is deliberate. Whatever the hand's own protocol turns out to be, it
lands in that one node, and nothing above it (this plugin, the controllers, MoveIt, the skills)
changes between simulation and the robot.

Separate from `g1_hardware_interface` on purpose. The hand is a different device with different
topics and its own control authority, and one component per hand keeps a hand fault from taking
the arms down with it.

## Interfaces

| Interface | Kind | Source |
|---|---|---|
| `position` | command | ramped, then clamped to the URDF limits |
| `position`, `velocity`, `effort` | state | `motor_state[i].q` / `.dq` / `.tau_est` |

| Topic | Direction | Type | Parameter |
|---|---|---|---|
| `/rh56e2/<side>/cmd` | out | `unitree_hg/msg/HandCmd` | `cmd_topic` |
| `/rh56e2/<side>/state` | in | `unitree_hg/msg/HandState` | `state_topic` |

Both at sensor QoS. `<side>` comes from the `side` parameter and must be `left` or `right`.

## Parameters

Values live in `g1_description/config/rh56e2_params.yaml`.

| Parameter | Default | Meaning |
|---|---|---|
| `side` | none | `left` or `right`. Picks the default topics and the joint prefix. Required. |
| `kp`, `kd` | 1.5, 0.2 | **Placeholders.** The Dex3's gains; nothing equivalent exists for this hand yet. |
| `command_publish_rate` | 100.0 | Hz. |
| `max_joint_velocity_rad_s` | 3.0 | Slew clamp on the commanded position. |
| `state_timeout_ms` | 200.0 | State older than this blocks activation. |
| `cmd_topic`, `state_topic` | `/rh56e2/<side>/...` | The topic pair. |

## Things the wire format will punish you for

**`motor_cmd` is an unbounded sequence, not a fixed array.** Publish it unresized and DDS accepts
the message while nothing moves. `on_configure` resizes once and `publish()` resizes again.

**Joint order is positional.** Six driven joints per hand, in this order, identical for both:
`thumb_1`, `thumb_2`, `index_1`, `middle_1`, `ring_1`, `little_1`. `on_init` refuses to start if
the declared joints disagree, because the failure is otherwise a hand that closes the wrong
fingers. The order also appears in the xacro, `controllers.yaml`, `moveit_controllers.yaml` and
the SRDF group, and `g1_moveit_config`'s drift test compares them.

**Only the driven joints exist here.** The hand has twelve non-fixed joints. The other six are
URDF mimics of these (`thumb_3`, `thumb_4`, and the second joint of each finger); they are never
commanded and not declared to `ros2_control`.

**There is no blend weight.** Unlike `rt/arm_sdk`, the first publish takes full authority
immediately. That is why this component seeds its command from the measured position on activate
and slews toward the target: nothing upstream will soften a step for you.

**Timeout protection** is armed only on release. While driving, the controller is the heartbeat,
and arming it would stop the fingers a second after any hiccup in the control loop.

## In simulation

`unitree_mujoco` answers the same two topics, so this component is not swapped out for sim. See
`workspace/vendor/unitree_mujoco/dex3_handler.cc` (the file keeps its name for now) for what the
responder does.

The simulated hand collides: every hand body has a convex-hull collision geom (patch 003), so it
stops at walls and holds objects by friction between the finger hulls and the object. The
contact is only as good as `g1_description/meshes/rh56e2/`: with the checked-in placeholder STLs
the hulls are tiny tetrahedra and the hand effectively does not collide. `grasp_weld` can still
weld a fully closed hand to its object as an assist; with contact on, a hand closing on a large
object stops on its surface and may never reach the weld's closure threshold, so the grip is
then friction alone. MoveIt still models a held object by attaching it in its planning scene.

## Not yet verified

Everything about the real hand. The gains, the state rate, the effort limits and whether the
vendor's joint ranges survive contact with the mechanism are all unmeasured, and there is no
driver to measure them with.
