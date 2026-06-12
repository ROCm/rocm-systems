#!/usr/bin/env bash
# Build and run GIN_ANVIL device-header unit tests (rccl-UnitTestsFixtures).
# Repo layout: this script lives next to projects/rccl (e.g. rocm-systems.git root).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${REPO_ROOT}/projects/rccl/CMakeLists.txt" ]]; then
  echo "error: expected ${REPO_ROOT}/projects/rccl/CMakeLists.txt (wrong script location?)" >&2
  exit 1
fi

if [[ -n "${ROCSHMEM_SOURCE_DIR:-}" ]]; then
  RS_DIR="$(cd "${ROCSHMEM_SOURCE_DIR}" && pwd)"
else
  RS_DIR="$(cd "${REPO_ROOT}/projects/rocshmem" && pwd)"
fi
if [[ ! -f "${RS_DIR}/src/sdma/anvil_device.hpp" ]]; then
  echo "error: rocshmem SDMA sources missing under ${RS_DIR} (expected src/sdma/anvil_device.hpp)" >&2
  echo "hint: export ROCSHMEM_SOURCE_DIR to the rocshmem repo root (not .../rocshmem/src)" >&2
  exit 1
fi

RCCL_BUILD="${REPO_ROOT}/projects/rccl/build"
cmake -S "${REPO_ROOT}/projects/rccl" -B "${RCCL_BUILD}" \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx950 \
  -DBUILD_TESTS=ON \
  -DENABLE_ROCSHMEM_GIN=ON \
  -DROCSHMEM_SOURCE_DIR="${RS_DIR}"
cmake --build "${RCCL_BUILD}" --target rccl-UnitTestsFixtures -j"$(nproc)"

RCCL_UTFX=""
for d in "${RCCL_BUILD}/test" "${RCCL_BUILD}/test/Release" "${RCCL_BUILD}/test/Debug" "${RCCL_BUILD}/test/RelWithDebInfo"; do
  if [[ -x "${d}/rccl-UnitTestsFixtures" ]]; then
    RCCL_UTFX="${d}/rccl-UnitTestsFixtures"
    break
  fi
done
if [[ -z "${RCCL_UTFX}" ]]; then
  RCCL_UTFX="$(find "${RCCL_BUILD}" -maxdepth 6 -type f -name rccl-UnitTestsFixtures 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${RCCL_UTFX}" || ! -x "${RCCL_UTFX}" ]]; then
  echo "error: rccl-UnitTestsFixtures not found or not executable under ${RCCL_BUILD}" >&2
  exit 1
fi

# Optional: HIP_VISIBLE_DEVICES=0 if you need to pin a GPU.
exec "${RCCL_UTFX}" --gtest_filter='GinAnvilDeviceTest.*'
