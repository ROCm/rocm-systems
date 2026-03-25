# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

check_rocprofsys_disable_examples("unified-memory" _UNIFIED_MEMORY_DISABLED)
if(_UNIFIED_MEMORY_DISABLED)
    return()
endif()

if(NOT HIPCC_EXECUTABLE)
    rocprofiler_systems_message(AUTHOR_WARNING "hipcc not found. Cannot build unified-memory target.")
    return()
endif()

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC)
    if(
        CMAKE_CXX_COMPILER STREQUAL HIPCC_EXECUTABLE
        OR "${CMAKE_CXX_COMPILER}" MATCHES "hipcc"
    )
        set(CMAKE_CXX_COMPILER_IS_HIPCC 1 CACHE BOOL "HIP compiler")
    endif()
endif()

set(HPC_EXAMPLE_NAME "unified-memory")
add_executable(${HPC_EXAMPLE_NAME} test_unified_memory.cpp)

target_compile_options(${HPC_EXAMPLE_NAME} PRIVATE -O2 -g)

foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
    target_compile_options(${HPC_EXAMPLE_NAME} PRIVATE --offload-arch=${arch})
    target_link_options(${HPC_EXAMPLE_NAME} PRIVATE --offload-arch=${arch})
endforeach()

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC)
    rocprofiler_systems_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET ${HPC_EXAMPLE_NAME})
endif()

rocprofiler_systems_install_hpc_example(${HPC_EXAMPLE_NAME})
