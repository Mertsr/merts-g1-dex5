# RH56E2 meshes

The hand descriptions (`urdf/rh56e2_left.urdf.xacro`, `urdf/rh56e2_right.urdf.xacro`) reference
their meshes as `package://g1_description/meshes/rh56e2/{left,right}/<name>.STL`. The STL files
come from the vendor's SolidWorks export (`RH56E2_L_2025_9_10`, `RH56E2_R_2025_9_11`) and are
checked in. They used to be placeholder tetrahedra, kept out of the repo, with the real ones
dropped in by hand -- which a fresh copy of the package silently replaced with the placeholders
again (the hands vanished from RViz). The inertials in rh56e2_left/right.urdf.xacro are the
vendor's own values and never depended on these files.

Put the contents of each export's `meshes/` folder here:

| Vendor package | Goes in |
|---|---|
| `RH56E2_L_2025_9_10/meshes/*.STL` | `left/` |
| `RH56E2_R_2025_9_11/meshes/*.STL` | `right/` |

Both exports name their base mesh `base_link.STL`, which is why the two sides have separate
folders. Expected files per side: `base_link`, `{side}_plam_1`, `{side}_plam_2`,
`{side}_plam_force_sensor`, `{side}_thumb_1` to `{side}_thumb_4`,
`{side}_thumb_force_sensor_1` to `_4`, and for each of `index`, `middle`, `ring`, `little`:
`_1`, `_2`, `_force_sensor_1`, `_force_sensor_2`, `_force_sensor_3`.

The finger meshes double as collision geometry. If MoveIt planning gets slow, simplify them
before anything else; the G1's own collision meshes are the reason `collisions_updater` cannot
finish on this robot (see `g1_moveit_config/README.md`).
