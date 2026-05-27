# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compile HIP device kernels into host+device ELF objects for simulator testing.
#
# The resulting .o files are standard ELF objects containing a .hip_fatbin
# section with the bundled AMDGPU device code. The code object loader
# (Executable) extracts and parses the device ELF from this section.
#
# Requires an ROCm installation with amdclang++. When the compiler is not
# found the kernel targets are silently skipped and tests that depend on them
# are disabled at runtime via the HAS_DEVICE_KERNELS compile definition.

# Prefer the compiler from the ROCm tree selected by ROCM_PATH. Some systems
# also expose /usr/bin/amdclang++ through alternatives; deriving --rocm-path
# from that wrapper gives /usr, which mixes headers and device libraries from
# different installs.
find_program(
    AMDCLANG
    amdclang++
    HINTS
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES bin
    DOC "Path to the ROCm amdclang++ compiler"
)

if(NOT AMDCLANG)
    message(
        STATUS
        "amdclang++ not found - device kernel tests will be disabled"
    )
    return()
endif()

if(DEFINED SKIP_DEVICE_KERNELS AND SKIP_DEVICE_KERNELS)
    message(
        STATUS
        "SKIP_DEVICE_KERNELS set - device kernel tests will be disabled"
    )
    return()
endif()

message(STATUS "Found amdclang++: ${AMDCLANG}")

set(KERNEL_OUTPUT_DIR ${CMAKE_BINARY_DIR}/kernels)
file(MAKE_DIRECTORY ${KERNEL_OUTPUT_DIR})

# Locate the ROCm installation root for --rocm-path.
get_filename_component(ROCM_PATH "${AMDCLANG}" DIRECTORY)
get_filename_component(ROCM_PATH "${ROCM_PATH}" DIRECTORY)

include(rj_add_device_kernel)

rj_add_device_kernel(matmul_naive gfx950)
rj_add_device_kernel(matmul_mfma gfx950)
rj_add_device_kernel(matmul_mfma_16x16 gfx950)
rj_add_device_kernel(matmul_tiled gfx950)
rj_add_device_kernel(vector_add gfx950)

# Umbrella target for all device kernels.
add_custom_target(
    device_kernels
    DEPENDS
        kernel_matmul_naive
        kernel_matmul_mfma
        kernel_matmul_mfma_16x16
        kernel_matmul_tiled
        kernel_vector_add
)

# Export variables to the parent scope so the test executable can use them.
set(HAS_DEVICE_KERNELS TRUE PARENT_SCOPE)
set(KERNEL_OUTPUT_DIR ${KERNEL_OUTPUT_DIR} PARENT_SCOPE)
