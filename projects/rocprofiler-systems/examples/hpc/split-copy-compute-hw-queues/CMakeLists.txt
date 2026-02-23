# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
enable_language(HIP)

# Compiler flags
set(EXAMPLE_COMPILE_FLAGS -O3 -g -DNDEBUG)
# Supress certain warnings
list(APPEND EXAMPLE_COMPILE_FLAGS -Wno-unused-result -Wno-vla-cxx-extension)
# Example specific
list(APPEND EXAMPLE_COMPILE_FLAGS -munsafe-fp-atomics)

set(IS_INSTINCT_APU FALSE)
foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
    rocprofiler_systems_lookup_gfx(${arch} _GPU_CATEGORIES)
    if("instinct" IN_LIST _GPU_CATEGORIES AND "apu" IN_LIST _GPU_CATEGORIES)
        set(IS_INSTINCT_APU TRUE)
    endif()
endforeach()

if(IS_INSTINCT_APU)
    list(APPEND EXAMPLE_COMPILE_FLAGS -DMI300A_RUN)
endif()

set(HPC_EXAMPLE_NAME "split-copy-compute-hw-queues")
add_executable(${HPC_EXAMPLE_NAME} compute_comm_overlap.hip)
target_compile_options(${HPC_EXAMPLE_NAME} PRIVATE ${EXAMPLE_COMPILE_FLAGS})

rocprofiler_systems_install_hpc_example(${HPC_EXAMPLE_NAME})
