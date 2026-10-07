# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# An explicitly empty path disables the environment and /opt/rocm fallbacks.
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(ROCM_PATH
            "$ENV{ROCM_PATH}"
            CACHE PATH
            "Path to the ROCm installation for tests"
        )
    elseif(IS_DIRECTORY "/opt/rocm")
        set(ROCM_PATH
            "/opt/rocm"
            CACHE PATH
            "Path to the ROCm installation for tests"
        )
    endif()
endif()

if(NOT "${ROCM_PATH}" STREQUAL "" AND NOT IS_DIRECTORY "${ROCM_PATH}")
    message(FATAL_ERROR "ROCM_PATH must be set to a valid directory")
endif()

# Keep the compiler under ROCM_PATH to avoid mixing SDKs through system wrappers.
find_program(
    AMDCXX
    NAMES amdclang++
    PATHS "${ROCM_PATH}"
    PATH_SUFFIXES lib/llvm/bin bin
    NO_DEFAULT_PATH
    NO_CACHE
)

# Share the capability result with kernel fixtures and host-side HIP tests.
set(RJ_HAS_HIP_KERNEL_TOOLCHAIN FALSE)
if(AMDCXX)
    include(rj_check_hip_kernel_toolchain)
endif()
