# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

check_rocprofsys_disable_examples("split-copy-compute-hw-queues" _SPLIT_COPY_COMPUTE_DISABLED)
if(_SPLIT_COPY_COMPUTE_DISABLED)
    return()
endif()

if(NOT hipcc_EXECUTABLE)
    rocprofiler_systems_message(AUTHOR_WARNING "hipcc not found, this is required for split-copy-compute-hw-queues example")
    return()
endif()

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

set_source_files_properties(compute_comm_overlap.hip PROPERTIES LANGUAGE CXX)

set(HPC_EXAMPLE_NAME "split-copy-compute-hw-queues")
add_executable(${HPC_EXAMPLE_NAME} compute_comm_overlap.hip)
target_compile_options(${HPC_EXAMPLE_NAME} PRIVATE -xhip ${EXAMPLE_COMPILE_FLAGS})

foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
    target_compile_options(${HPC_EXAMPLE_NAME} PRIVATE --offload-arch=${arch})
    target_link_options(${HPC_EXAMPLE_NAME} PRIVATE --offload-arch=${arch})
endforeach()

rocprofiler_systems_custom_compilation(COMPILER ${hipcc_EXECUTABLE} TARGET ${HPC_EXAMPLE_NAME})

rocprofiler_systems_install_hpc_example(${HPC_EXAMPLE_NAME})
