#!/usr/bin/env bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run the corpus HIP file-limit matrix with the same configuration as CI.
# Required: ROCM_PATH, ROCJITSU_BUILD_DIR, ROCJITSU_CORPUS_DIR.
# Optional: ROCJITSU_SOURCE_DIR (defaults to this script's source tree).
# Additional arguments are passed to CTest, e.g. -R local.early -V.
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: ROCM_PATH=/path/to/sdk ROCJITSU_BUILD_DIR=/path/to/build \
ROCJITSU_CORPUS_DIR=/path/to/corpus bash $0 [CTest options]"
  echo "Build rocjitsu first. ASan/TSan clients use its cached C++ compiler."
  echo "Clang must support gfx950 HIP with ROCM_PATH; use RJ_SANITIZER_RUNTIME=AUTO or SHARED."
  echo "Build parallelism follows Ninja or CMAKE_BUILD_PARALLEL_LEVEL; CTest defaults to 4."
  exit 0
fi

: "${ROCM_PATH:?ROCM_PATH must be set}"
: "${ROCJITSU_BUILD_DIR:?ROCJITSU_BUILD_DIR must be set}"
: "${ROCJITSU_CORPUS_DIR:?ROCJITSU_CORPUS_DIR must be set}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(realpath "${ROCJITSU_SOURCE_DIR:-${script_dir}/../..}")"
build_dir="$(realpath "${ROCJITSU_BUILD_DIR}")"
corpus_dir="$(realpath "${ROCJITSU_CORPUS_DIR}")"
rocm_path="$(realpath "${ROCM_PATH}")"
cache="${build_dir}/CMakeCache.txt"
launcher="${build_dir}/tools/rocjitsu/rocjitsu"
config="${source_dir}/tests/race-detector/race_test_config.json"
if [[ ! -f "${cache}" || ! -x "${launcher}" ]]; then
  echo "Build rocjitsu in ROCJITSU_BUILD_DIR before running the file-limit suite." >&2
  exit 1
fi
if [[ ! -f "${corpus_dir}/corpus/runtime/file-limits/CMakeLists.txt" || ! -f "${config}" ]]; then
  echo "Missing file-limit corpus sources or matching rocjitsu config." >&2
  exit 1
fi
if ! command -v tini >/dev/null; then
  echo "Install tini or add it to PATH so detached memory keepers are reaped." >&2
  exit 1
fi

# Match CI's client instrumentation. UBSan-only rocjitsu builds use an
# uninstrumented HIP client; ASan and TSan builds need matching host runtimes.
client_sanitizer=""
if grep -Eq '^RJ_ENABLE_TSAN:BOOL=(1|ON|TRUE|YES)$' "${cache}"; then
  client_sanitizer=thread
elif grep -Eq '^RJ_ENABLE_ASAN:BOOL=(1|ON|TRUE|YES)$' "${cache}"; then
  client_sanitizer=address,undefined
fi
client_compiler_args=()
if [[ -n "${client_sanitizer}" ]]; then
  client_compiler="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "${cache}")"
  runtime_linkage="$(sed -n 's/^RJ_SANITIZER_RUNTIME:[^=]*=//p' "${cache}")"
  if [[ "${runtime_linkage^^}" != AUTO && "${runtime_linkage^^}" != SHARED ]]; then
    echo "ASan/TSan file-limit clients require rocjitsu built with RJ_SANITIZER_RUNTIME=AUTO or SHARED." >&2
    exit 1
  fi
  if [[ ! -x "${client_compiler}" ]] ||
     ! "${client_compiler}" --version | grep -qi clang ||
     ! "${client_compiler}" --print-targets | grep -q amdgcn; then
    echo "ASan/TSan file-limit clients require the cached CMAKE_CXX_COMPILER to be Clang with HIP/AMDGPU support: ${client_compiler}" >&2
    exit 1
  fi
  if ! "${client_compiler}" --rocm-path="${rocm_path}" --offload-arch=gfx950 \
       -x hip -fsyntax-only - </dev/null; then
    echo "The cached Clang cannot compile HIP for gfx950 with ROCM_PATH=${rocm_path}; use a compatible compiler/SDK pair." >&2
    exit 1
  fi
  # The launcher's preloaded sanitizer and the HIP client's linked runtime must
  # come from the same compiler, including system Clang's unsuffixed SONAMEs.
  client_compiler_args+=("-DCMAKE_HIP_COMPILER=${client_compiler}")
fi
file_limit_source_dir="$(realpath "${corpus_dir}/corpus/runtime/file-limits")"
file_limit_build_dir="${build_dir}/file-limit-corpus"
file_limit_cache="${file_limit_build_dir}/CMakeCache.txt"
refresh_cache=false
if [[ -f "${file_limit_cache}" ]]; then
  previous_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:[^=]*=//p' "${file_limit_cache}")"
  if [[ "${previous_source}" != "${file_limit_source_dir}" ]]; then
    refresh_cache=true
  fi
  if [[ -n "${client_sanitizer}" ]]; then
    previous_compiler="$(sed -n 's/^CMAKE_HIP_COMPILER:[^=]*=//p' "${file_limit_cache}")"
    if [[ "${previous_compiler}" != "${client_compiler}" ]]; then
      refresh_cache=true
    fi
  fi
fi
echo "Running gfx950 HIP file-limit tests (client sanitizer: ${client_sanitizer:-none})"
configure=(
  cmake -S "${file_limit_source_dir}"
  -B "${file_limit_build_dir}" -G Ninja
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DROCM_PATH="${rocm_path}"
  -DCMAKE_HIP_ARCHITECTURES=gfx950
  "${client_compiler_args[@]}"
  -DCMAKE_BUILD_RPATH="${rocm_path}/lib"
  -DFILE_LIMIT_SANITIZER="${client_sanitizer}"
  -DROCJITSU="${launcher}"
  -DROCJITSU_CONFIG="${config}"
)
if [[ "${refresh_cache}" == true ]]; then
  # CMake rejects source changes and clears settings when switching compilers.
  # Start fresh so the requested source, launcher, and instrumentation take effect.
  configure+=(--fresh)
fi
"${configure[@]}"
cmake --build "${file_limit_build_dir}"
exec tini -s -- ctest --test-dir "${file_limit_build_dir}" \
  --output-on-failure --no-tests=error -j 4 "$@"
