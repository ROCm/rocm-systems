# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(rocprofiler-systems-rocshmem LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

# Determine warning level: downgrade to STATUS during CI to avoid spurious warnings
if("${CMAKE_PROJECT_NAME}" STREQUAL "rocprofiler-systems" AND "$ENV{ROCPROFSYS_CI}")
    set(_MSG_TYPE STATUS)
else()
    set(_MSG_TYPE AUTHOR_WARNING)
endif()

# HIP runtime is required for hipStream_t, hipStreamCreate/Synchronize/Destroy
find_package(hip HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm QUIET)
if(NOT hip_FOUND)
    message(${_MSG_TYPE} "rocshmem skipped: HIP not found.")
    return()
endif()

# rocSHMEM provides the roc::rocshmem imported target. When built as part of
# the rocm-systems monorepo the target is already defined; for standalone
# builds we attempt a find_package.
if(NOT TARGET roc::rocshmem)
    find_package(rocshmem HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm QUIET)
endif()
if(NOT TARGET roc::rocshmem)
    message(${_MSG_TYPE} "rocshmem skipped: rocSHMEM (roc::rocshmem) not found.")
    return()
endif()

# MPI is optional at the source level — rocshmem_init() handles MPI setup
# internally — but linking MPI ensures correct symbol resolution when the
# demo is launched via mpirun.
find_package(MPI QUIET)

enable_language(HIP)

add_executable(rocshmem rocshmem.cpp)
set_source_files_properties(rocshmem.cpp PROPERTIES LANGUAGE HIP)

set_target_properties(
    rocshmem
    PROPERTIES
        HIP_STANDARD          17
        HIP_STANDARD_REQUIRED ON
        CXX_STANDARD          17
        CXX_STANDARD_REQUIRED ON
)

target_link_libraries(
    rocshmem
    PRIVATE
        roc::rocshmem
        hip::host
        $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
        $<TARGET_NAME_IF_EXISTS:rocprofiler-systems::rocprofiler-systems-compile-options>
)

# Export the binary path so CMake test configuration files (e.g.
# rocprof-sys-rocshmem-tests.cmake) can locate the executable without
# hard-coding a path.
set(ROCPROFSYS_ROCSHMEM_DEMO_EXE
    "$<TARGET_FILE:rocshmem>"
    CACHE INTERNAL
    "Path to rocshmem executable"
)

if(ROCPROFSYS_INSTALL_EXAMPLES)
    install(
        TARGETS rocshmem
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
        OPTIONAL
    )
endif()
