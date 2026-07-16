# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compile a .hip source into a host+device ELF object (.o).
#
# The -c flag produces a relocatable object with a .hip_fatbin section
# that contains the Clang offload bundle with the device code object for
# the specified GPU target. AMDCXX, ROCM_PATH, and KERNEL_OUTPUT_DIR
# must be set before including this module.
#
# Usage: rj_add_device_kernel(<name> <offload_arch> [OUTPUT_NAME <output_name>])
#   name         - base name of the .hip source (without extension)
#   offload_arch - GPU target (e.g. gfx942, gfx950)
#   OUTPUT_NAME  - optional; basename of the output .o (default: ${name}).
#                  Use when the same source is compiled for multiple targets
#                  so each build lands in a distinct .o file. (Multi-target
#                  fat binaries via repeated --offload-arch flags would be
#                  more elegant but the Executable fat-binary loader has a
#                  pre-existing single-bundle assumption — see
#                  executable.cpp::load_hip_fatbin.)
function(rj_add_device_kernel name offload_arch)
    set(oneValueArgs OUTPUT_NAME)
    cmake_parse_arguments(RJ_KERNEL "" "${oneValueArgs}" "" ${ARGN})

    set(output_name "${name}")
    if(RJ_KERNEL_OUTPUT_NAME)
        set(output_name "${RJ_KERNEL_OUTPUT_NAME}")
    endif()

    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${name}.hip)
    set(out ${KERNEL_OUTPUT_DIR}/${output_name}.o)

    add_custom_command(
        OUTPUT ${out}
        COMMAND
            ${AMDCXX} -x hip --offload-arch=${offload_arch}
            --rocm-path=${ROCM_PATH} -fPIC -c -O2 -o ${out} ${src}
        DEPENDS ${src}
        COMMENT "Compiling device kernel: ${output_name} (${offload_arch})"
    )

    # Per-kernel target so dependents can reference it.
    add_custom_target(kernel_${output_name} DEPENDS ${out})
endfunction()

# Compile a .hip device-function source into a standalone device ELF (.hsaco).
#
# Unlike rj_add_device_kernel above (which compiles a __global__ kernel into a
# host object carrying a .hip_fatbin), this is for a lone __device__ function
# with no kernel. The normal fatbin path drops such a function, so this:
#   1. compiles device-only with -fgpu-rdc into a Clang offload bundle, then
#   2. unbundles the device code object into a raw device ELF that Executable /
#      AmdGpuCodeObject can load directly.
#
# AMDCXX, ROCM_PATH, CLANG_OFFLOAD_BUNDLER, and KERNEL_OUTPUT_DIR must be set
# before calling this function.
#
# Usage: rj_add_probe_object(<name> <offload_arch> [OUTPUT_NAME <output_name>])
#   name         - base name of the .hip source (without extension)
#   offload_arch - GPU target (e.g. gfx90a)
#   OUTPUT_NAME  - optional; basename of the output .hsaco (default: ${name}).
#
# Creates a custom target probe_${output_name} producing
# ${KERNEL_OUTPUT_DIR}/${output_name}.hsaco.
function(rj_add_probe_object name offload_arch)
    set(oneValueArgs OUTPUT_NAME)
    cmake_parse_arguments(RJ_PROBE "" "${oneValueArgs}" "" ${ARGN})

    set(output_name "${name}")
    if(RJ_PROBE_OUTPUT_NAME)
        set(output_name "${RJ_PROBE_OUTPUT_NAME}")
    endif()

    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${name}.hip)
    set(bundle ${KERNEL_OUTPUT_DIR}/${output_name}.bundle)
    set(hsaco ${KERNEL_OUTPUT_DIR}/${output_name}.hsaco)

    # Step 1: device-only compile to a Clang offload bundle. -fgpu-rdc keeps the
    # used/noinline device function from being internalized away.
    add_custom_command(
        OUTPUT ${bundle}
        COMMAND
            ${AMDCXX} -x hip --offload-arch=${offload_arch}
            --rocm-path=${ROCM_PATH} -fgpu-rdc --cuda-device-only -O2 -o
            ${bundle} ${src}
        DEPENDS ${src}
        COMMENT
            "Compiling probe object (device-only): ${output_name} (${offload_arch})"
    )

    # Step 2: unbundle the device code object into a raw device ELF. The bundler
    # matches the device entry by Target ID prefix
    add_custom_command(
        OUTPUT ${hsaco}
        COMMAND
            ${CLANG_OFFLOAD_BUNDLER} --type=o --unbundle
            --targets=hipv4-amdgcn-amd-amdhsa--${offload_arch} --input=${bundle}
            --output=${hsaco}
        DEPENDS ${bundle}
        COMMENT "Unbundling probe object: ${output_name} (${offload_arch})"
    )

    add_custom_target(probe_${output_name} DEPENDS ${hsaco})
endfunction()
