# g1_hand_interface

Two independent `ros2_control` `SystemInterface` plugins, each in its own library. `ament_cmake`,
C++20.

| Plugin | Library | Status |
|---|---|---|
| `g1_hand_interface/G1Dex5SimSystem` | `g1_dex5_sim_system` | **Used.** Simulated Dex5 hand. No device, no DDS, no SDK link. |
| `g1_hand_interface/G1Dex3System` | `g1_dex3_system` | **Legacy, not wired in.** One Dex3-1 hand over `unitree_sdk2`. Nothing in the xacro, controller config or launch/orchestration path references it. |

## G1Dex5SimSystem

The first Dex5 control milestone: **simulation only.** One instance per hand (the xacro declares `G1Dex5SystemLeft` and
`G1Dex5SystemRight`). Each claims the six active joints of its hand, in this pinned order, and refuses to initialise on any other list (so the mimic
joints can never be claimed):

`left_thumb_1_joint`, `left_thumb_2_joint`, `left_index_1_joint`, `left_middle_1_joint`,
`left_ring_1_joint`, `left_little_1_joint`; the right hand uses the same suffixes with `right_`.

| Interface | Kind | Behaviour |
|---|---|---|
| `position` | command | Clamped to the joint's `min`/`max`; a non-finite value holds the joint. |
| `position`, `velocity`, `effort` | state | Position approaches the command first-order, speed-capped, without overshoot. Velocity is the per-update position difference. Effort is always 0. |

`on_activate` seeds the command from the current state, so the first update moves nothing; the
state is always finite. `on_deactivate` zeroes the velocity and holds. The model advances only while
active, using the controller period, and caps one step at 0.1 s so a stalled loop cannot jump.
`Dex5SimModel` (`dex5_sim_model.hpp`) holds that logic with no ROS dependency.

Parameters, from `g1_description/config/dex5_params.yaml` through the xacro:

| Parameter | Default | Meaning |
|---|---|---|
| `side` | required | `left` or `right`; picks the joint prefix; the xacro declares both. |
| `min` / `max` (per joint) | required | Position limits in rad, copied from the URDF. |
| `max_joint_velocity_rad_s` | 3.0 | Speed cap. |
| `time_constant_s` | 0.1 (YAML sets 0.05) | Time constant of the approach. |

Select it from the robot description with `xacro g1_lowcmd.urdf.xacro dex5_backend:=sim` (the
default). A physical backend (Modbus/TCP, a MuJoCo hand model, touch sensors) does not exist here;
it would be another plugin and another `backends:` entry, behind the same joints, order and radians,
mapping to the device's own register order internally.

## Dex3 legacy plugin

`G1Dex3System`: the `ros2_control` `SystemInterface` for one Unitree Dex3-1 hand, over the hand's
own `unitree_sdk2` channels. One component per hand. `ament_cmake`, C++20.

```mermaid
flowchart LR
    HC["left/right_hand_controller"] -- "position" --> P["G1Dex3System<br/>one component per hand"]
    P -- "rt/dex3/side/cmd" --> H["the hand, or unitree_mujoco"]
    H -- "rt/dex3/side/state" --> P
```

Separate from `g1_hardware_interface` so a hand fault cannot take the arms down with it. Like that
component, it links the SDK's own CycloneDDS with `DT_RPATH` pinned at the SDK's `lib/`, so ROS's
CycloneDDS (same SONAME, different ABI) is never bound in its place. `domain_id` and
`network_interface` must match `g1_hardware_interface`'s, because `ChannelFactory` is per process
and only its first `Init` takes effect.

No xacro declares this component any more: the Dex3 joints are gone from the URDF, so it cannot
initialise against the current description, and `g1_description/config/dex3_params.yaml` was removed.
It is kept only so the Dex3 wire contract and its tests stay available. The rest of this section
describes it as it was.

## Interfaces

| Interface | Kind | Source |
|---|---|---|
| `position` | command | clamped to the joint limits, then slewed |
| `position`, `velocity`, `effort` | state | `motor_state[i].q` / `.dq` / `.tau_est` |

| Channel | Direction | Type |
|---|---|---|
| `rt/dex3/<side>/cmd` | out | `unitree_hg::msg::dds_::HandCmd_` |
| `rt/dex3/<side>/state` | in | `unitree_hg::msg::dds_::HandState_` (`state_topic`) |

These are SDK channels, not ROS topics, so they carry the SDK's own QoS and do not appear in
`ros2 topic list`.

## Parameters

These were passed from the since-removed `dex3_params.yaml`. The defaults below are the component's
own; the old YAML matched them except for `kp` (2.3) and `kd` (0.25).

| Parameter | Default | Meaning |
|---|---|---|
| `side` | required | `left` or `right`. Picks the channels and the joint prefix. |
| `domain_id` | required | SDK DDS domain. Must match `g1_hardware_interface`'s. |
| `network_interface` | `""` | Must stay empty: a non-empty value makes the SDK discard `CYCLONEDDS_URI`. |
| `kp` / `kd` | 1.5 / 0.2 | Finger PD gains, sent in every driven frame. The YAML sets 2.3 / 0.25: 2.3 lets a thumb stalled about 0.6 rad short of its target press at the joint's 1.4 Nm rating. |
| `command_publish_rate` | 100.0 | Hz. What Unitree's own teleop uses. |
| `max_joint_velocity_rad_s` | 3.0 | Slew clamp on the commanded position. |
| `state_timeout_ms` | 200.0 | State older than this errors the component while active. |
| `state_topic` | `rt/dex3/<side>/state` | State channel. The robot also carries a lower-rate `rt/lf/dex3/<side>/state`, and Unitree's own code disagrees about which to read. The full-rate one is the default because this is a control loop with a freshness gate. The YAML gives a prefix and suffix that the xacro joins around the side. |
| `min` / `max` | required, per joint | Position limits in rad. Commands are clamped to them. |

`kp`, `kd`, `command_publish_rate`, `max_joint_velocity_rad_s` and `state_timeout_ms` must be
positive, or `on_init` refuses. Activation waits up to 5 s for a first `HandState` and fails without
one.

## Wire format notes

- `motor_cmd` is an unbounded sequence, not a fixed array. Unresized, DDS accepts the message and
  nothing moves. `on_init` resizes the preallocated frame once, so the write path never allocates.
- Joint order is positional and matches the URDF: thumb_0, thumb_1, thumb_2, middle_0, middle_1,
  index_0, index_1, for both hands. `on_init` refuses any other order, since the failure is a hand
  that closes the wrong fingers. Unitree's `Dex3_1_Right_JointIndex` enum lists index before
  middle; do not copy it.
- There is no blend weight: the first write takes full authority. The component seeds its command
  from the measured position on activate and slews toward the target.
- The motors' own timeout is armed only in the release frame. While driving, the controller is the
  heartbeat, and arming it would stop the fingers a second after any hiccup in the control loop.
- Limits are the URDF's, which match Unitree's published spec. Their SDK example allows a wider
  `thumb_1` (0.724 rad against 0.611), so the conservative pair wins.

## In simulation (Dex3, legacy)

The component speaks Unitree's published Dex3 contract, and `unitree_mujoco` answers the same two
channels, so it is not swapped out for sim. The responder is the simulator's `dex3_handler.cc`
(`workspace/vendor/unitree_mujoco/simulate/src`), on the finger joints its G1 model adds. It runs
the hardware's PD from the `kp` and `kd` in the command and clamps to the URDF's effort limits.

- The fingers are driven through `qfrc_applied`, not MuJoCo actuators: the vendored SDK bridge
  sizes itself from the actuator count and indexes a fixed 35-slot `LowCmd`, which 29 body motors
  plus 14 fingers would overrun.
- `status = Lock` holds the finger where it is rather than going limp, as on hardware. The same
  applies before any command arrives, and one second after the last one.
- Finger contact exists only where a scene opts in. The palm and fingers have collision primitives
  on their own contact bit (`contype` 2), so only the props and tables a scene marks `contype` 3
  can touch them.

## Tests

```bash
./scripts/manage.sh test g1_hand_interface
```

- `test_dex5_sim_model`: init from a valid state, smooth non-overshooting approach, speed cap, limit
  clamping, deterministic velocity, finite state for non-finite commands and bad periods.
- `test_dex5_sim_plugin`: `G1Dex5SimSystem` resolves through pluginlib, the path
  `controller_manager` uses.
- `test_dex5_sim_isolation`: the Dex5 sources and startup path never name `G1Dex3System`, the SDK or
  a DDS channel.
- `test_wire_contract`: the legacy Dex3 mode byte, joint order and frames, plus its plugin lookup.

`g1_description`'s `test_lowcmd_xacro` pins each component's six joints and that no mimic joint
is claimed. `g1_moveit_config`'s `test_moveit_lowcmd` runs both hands end to end
(`./scripts/manage.sh test --sim g1_moveit_config`); that needs the simulator for the body and was
not run when this was written.
