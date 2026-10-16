#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Compile the HRR capture sources, and nothing else, for CodeQL to trace.
#
# Capture is compiled into libamdhip64, so building it the usual way builds
# the whole HIP runtime and CodeQL would scan all of it. This compiles only the
# .cpp files in projects/clr/hipamd/src/hrr, with the flags amdhip64 gives
# them on Linux and HIP_HRR_CAPTURE_ENABLED, against the ROCm headers in
# ROCM_PATH. The objects are thrown away: CodeQL only needs to see the compiler
# run.
#
# Usage: hrr_capture_codeql_build.sh [REPO_ROOT]   (default: current directory)
set -euo pipefail

ROOT="$(cd "${1:-.}" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
CXX="${CXX:-${ROCM_PATH}/lib/llvm/bin/clang++}"
CLR="${ROOT}/projects/clr"
OUT="$(mktemp -d)"
trap 'rm -rf "${OUT}"' EXIT

# The definitions and include directories amdhip64 and rocclr give their
# sources in a Linux shared-library build (hipamd/src/CMakeLists.txt,
# rocclr/cmake/ROCclr*.cmake and FindAMD_OPENCL.cmake).
flags=(
  -std=c++17 -fPIC -O0 -w
  -DHIP_HRR_CAPTURE_ENABLED
  -D__HIP_PLATFORM_AMD__ -DHIP_ABI_IMPL -DUSE_PROF_API=1
  -DATI_OS_LINUX -DLITTLEENDIAN_CPU -DWITH_HSA_DEVICE -DCOMGR_DYN_DLL
  -DHAVE_CL2_HPP -DOPENCL_MAJOR=2 -DOPENCL_MINOR=1 -DOPENCL_C_MAJOR=2
  -DOPENCL_C_MINOR=0 -DCL_TARGET_OPENCL_VERSION=220
  -DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS
  -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS
  -I"${CLR}/hipamd/src/hrr"
  -I"${ROOT}/projects/hrr/include"
  -I"${CLR}/hipamd/src"
  -I"${CLR}/hipamd/include"
  -I"${ROOT}/projects/hip/include"
  -I"${CLR}"
  -I"${CLR}/rocclr"
  -I"${CLR}/rocclr/compiler/lib"
  -I"${CLR}/rocclr/compiler/lib/include"
  -I"${CLR}/rocclr/compiler/lib/backends/common"
  -I"${CLR}/rocclr/device"
  -I"${CLR}/rocclr/elf"
  -I"${CLR}/rocclr/include"
  -I"${CLR}/opencl/khronos/headers/opencl2.2/CL"
  -I"${CLR}/opencl/khronos/headers/opencl2.2"
  -I"${CLR}/opencl/khronos/headers"
  -I"${CLR}/opencl/khronos"
  -I"${CLR}/opencl"
  -I"${CLR}/opencl/amdocl"
  -I"${ROCM_PATH}/include"
  -I"${ROCM_PATH}/include/hsa"
)

sources=("${CLR}"/hipamd/src/hrr/*.cpp)
# A new file is picked up by the glob; a missing directory or an
# empty glob must fail rather than trace nothing.
[[ -e "${sources[0]}" ]] || { echo "No capture sources under ${CLR}/hipamd/src/hrr" >&2; exit 1; }

for src in "${sources[@]}"; do
  echo "compile ${src#"${ROOT}/"}"
  "${CXX}" "${flags[@]}" -c "${src}" -o "${OUT}/$(basename "${src}").o"
done
echo "${#sources[@]} capture source(s) compiled."
