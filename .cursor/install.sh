#!/usr/bin/env bash
# Cloud Agent install script for the ROCm Systems super-repo.
#
# Scope: this prepares the GPU-independent developer experience that can run on a
# CPU-only Cloud Agent VM:
#   * repository-wide formatting / lint tooling (pre-commit, ruff), and
#   * the rocprofiler-compute Python "analyze" workflow together with its
#     CPU-runnable analyze test suites (which operate on pre-recorded workload
#     data checked into the repo).
#
# Building the C/C++/HIP components of the super-repo requires AMD GPUs and a
# ROCm toolchain, which are not available in this environment, so those builds
# are intentionally out of scope here.
#
# The script is idempotent: it can be re-run safely and converges to the same
# state.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RPC="projects/rocprofiler-compute"

# 1. System packages. Mirrors the rocprofiler-compute CI baseline dependencies.
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git \
  python3 \
  python3-pip \
  cmake \
  build-essential \
  libdw-dev \
  pkg-config

# 2. Vendored dependency required for rocprofiler-compute to import/run.
#    Only this submodule is needed for the analyze workflow; the many other
#    submodules are for the GPU C++ builds and are left uninitialized.
git submodule update --init "$RPC/src/vendored/pyyaml"

# 3. Python dependencies (analyze-mode runtime + test deps) plus repo-wide dev
#    tooling. --break-system-packages is required on Ubuntu 24.04's externally
#    managed Python; --ignore-installed avoids failed uninstalls of distro
#    packages (e.g. PyYAML, blinker) that ship without pip RECORD files.
sudo -H python3 -m pip install --break-system-packages --ignore-installed \
  -r "$RPC/requirements.txt" \
  -r "$RPC/requirements-test.txt" \
  pre-commit \
  ruff

echo "Cloud Agent environment install complete."
