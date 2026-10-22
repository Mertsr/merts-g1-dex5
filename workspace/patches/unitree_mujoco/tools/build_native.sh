#!/usr/bin/env bash
# Builds the grove-g1 unitree_mujoco (`simulate` app) on a bare-metal ROS 2 Humble machine,
# i.e. without the devcontainer. It mirrors the "unitree_mujoco" stage of .devcontainer/Dockerfile
# exactly -- same pinned SHA, same patches in the same order, same file copy -- but against
# an output directory you choose and without touching apt.
#
# Prerequisites this script does NOT install (the Dockerfile's earlier stages do):
#   - unitree_sdk2, built and installed via `cmake --install` (so <prefix>/lib/cmake has its
#     config file); default prefix assumed is /opt/unitree_robotics, override with
#     UNITREE_SDK2_DIR
#   - A MuJoCo release tarball, EXTRACTED (not built) -- unitree_mujoco's own CMakeLists.txt
#     expects it as a plain subdirectory "mujoco/" under simulate/ with mujoco/include,
#     mujoco/simulate and mujoco/lib inside it, not a system/cmake-installed MuJoCo. Point
#     MUJOCO_DIR at wherever you extracted mujoco-X.Y.Z-linux-*.tar.gz; this script copies it
#     in rather than symlinking, matching what the Dockerfile's `mv` does.
#   - nlohmann-json3-dev, yaml-cpp, glfw3, Boost (program_options), and a C++17 compiler + cmake
# If any of those are missing, the cmake/build step below fails with a normal "package not
# found" error naming the missing one; this script does not try to guess your package manager.
#
# Usage:
#   UNITREE_SDK2_DIR=/opt/unitree_robotics MUJOCO_DIR=/opt/unitree_robotics/mujoco-3.1.6 \
#   ./build_native.sh /path/to/output/unitree_mujoco
#
# What it does, in order (identical to the Dockerfile's unitree_mujoco stage):
#   1. Clones unitree_mujoco at the pinned SHA into OUT_DIR.
#   2. Applies every workspace/patches/unitree_mujoco/*.patch in filename order, failing closed
#      on the first one that does not apply (git apply --check first, exactly like the image build).
#   3. Copies workspace/vendor/unitree_mujoco/*.{h,cc} over simulate/src/ -- the real logic the
#      patches only register.
#   4. Copies workspace/src/g1_description/meshes/rh56e2/{left,right} into
#      unitree_robots/g1/meshes/rh56e2/, so patch 003's `file="rh56e2/<side>/*.STL"` references
#      resolve. Placeholder STLs are enough to compile; replace them with the vendor's before
#      trusting the sim's grasp geometry.
#   5. Runs cmake + build for the `unitree_mujoco` target only (matches the Dockerfile: jstest
#      fails against the current SDK and isn't built here either).
#
# This script is idempotent only in the sense that it refuses to run twice into the same
# OUT_DIR -- rerunning after a patch edit means deleting OUT_DIR and starting over, same as a
# fresh image build would.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PATCH_DIR="$REPO_ROOT/workspace/patches/unitree_mujoco"
VENDOR_SRC="$REPO_ROOT/workspace/vendor/unitree_mujoco"
MESH_SRC="$REPO_ROOT/workspace/src/g1_description/meshes/rh56e2"
UNITREE_MUJOCO_SHA="ae6a8403e272733e9996ef59990880330496177f"

OUT_DIR="${1:?usage: build_native.sh <output-dir> (will be created; must not already exist)}"
UNITREE_SDK2_DIR="${UNITREE_SDK2_DIR:-/opt/unitree_robotics}"
MUJOCO_DIR="${MUJOCO_DIR:-}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

if [ -e "$OUT_DIR" ]; then
    echo "refusing to run: $OUT_DIR already exists -- remove it or pick a fresh path" >&2
    exit 1
fi
if [ -z "$MUJOCO_DIR" ] || [ ! -d "$MUJOCO_DIR/include" ] || [ ! -d "$MUJOCO_DIR/simulate" ]; then
    echo "MUJOCO_DIR must point at an EXTRACTED MuJoCo release (needs include/ and simulate/ subdirs)" >&2
    exit 1
fi
for f in "$MESH_SRC/left" "$MESH_SRC/right"; do
    if ! find "$f" -maxdepth 1 -iname '*.STL' -print -quit | grep -q .; then
        echo "WARNING: no STL files in $f -- the model will not load until they are added" >&2
    fi
done

echo "cloning unitree_mujoco @ $UNITREE_MUJOCO_SHA into $OUT_DIR"
git init -q "$OUT_DIR"
cd "$OUT_DIR"
git remote add origin https://github.com/unitreerobotics/unitree_mujoco.git
git fetch --depth 1 origin "$UNITREE_MUJOCO_SHA"
git checkout -q FETCH_HEAD

for p in "$PATCH_DIR"/*.patch; do
    [ -e "$p" ] || continue
    echo "applying $(basename "$p")"
    git apply --check "$p" || { echo "PATCH DOES NOT APPLY: $p" >&2; exit 1; }
    git apply "$p"
done

echo "copying handler sources into simulate/src"
cp "$VENDOR_SRC"/*.h "$VENDOR_SRC"/*.cc simulate/src/

echo "copying RH56E2 meshes into unitree_robots/g1/meshes/rh56e2"
mkdir -p unitree_robots/g1/meshes/rh56e2
cp -r "$MESH_SRC/left" "$MESH_SRC/right" unitree_robots/g1/meshes/rh56e2/

rm -rf .git

echo "placing MuJoCo under simulate/mujoco (unitree_mujoco's CMakeLists.txt expects it there,"
echo "not a system find_package)"
rm -rf simulate/mujoco
cp -r "$MUJOCO_DIR" simulate/mujoco
rm -rf simulate/mujoco/sample simulate/mujoco/bin simulate/mujoco/doc simulate/mujoco/model

echo "configuring and building the unitree_mujoco target ($BUILD_JOBS jobs)"
mkdir -p simulate/build
cmake -S simulate -B simulate/build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$UNITREE_SDK2_DIR/lib/cmake"
cmake --build simulate/build --target unitree_mujoco --parallel "$BUILD_JOBS"

echo
echo "done: $OUT_DIR/simulate/build/unitree_mujoco"
echo "run it from unitree_robots/g1 (the binary resolves mesh/scene paths relative to its cwd)"
