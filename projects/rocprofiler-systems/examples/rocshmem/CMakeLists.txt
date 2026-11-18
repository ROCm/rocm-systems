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
    message(${_MSG_TYPE} "rocshmem-test skipped: HIP not found.")
    return()
endif()

# rocSHMEM provides the roc::rocshmem imported target. When built as part of
# the rocm-systems monorepo the target is already defined; for standalone
# builds we attempt a find_package.
if(NOT TARGET roc::rocshmem)
    find_package(rocshmem HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm QUIET)
endif()

# Fallback for older rocSHMEM packages whose installed cmake config calls a bare
# find_dependency(NUMA) and cannot locate TheRock's sysdeps numa-config.cmake on
# its own. Only reached when the first attempt failed; newer rocSHMEM
# (rocm-systems#9583) resolves NUMA internally, so the first attempt already
# succeeds and this whole block is skipped, deferring entirely to rocSHMEM.
if(NOT TARGET roc::rocshmem)
    if(NOT NUMA_DIR)
        foreach(_rocm_root ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm)
            if(
                _rocm_root
                AND EXISTS "${_rocm_root}/lib/rocm_sysdeps/lib/cmake/NUMA/numa-config.cmake"
            )
                set(NUMA_DIR "${_rocm_root}/lib/rocm_sysdeps/lib/cmake/NUMA")
                break()
            endif()
        endforeach()
    endif()
    # Retry with whatever NUMA_DIR is now (user-provided or sysdeps-discovered).
    if(NUMA_DIR)
        find_package(rocshmem HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm QUIET)
    endif()
endif()

if(NOT TARGET roc::rocshmem)
    message(${_MSG_TYPE} "rocshmem-test skipped: rocSHMEM (roc::rocshmem) not found.")
    return()
endif()

# rocshmem.cpp exercises the full set of host-stream APIs introduced in rocSHMEM 3.6.0.
# Older releases (shipped with ROCm 7.2.x) only declare rocshmem_barrier_all_on_stream, so skip.
if(rocshmem_VERSION AND rocshmem_VERSION VERSION_LESS 3.6.0)
    message(
        ${_MSG_TYPE}
        "rocshmem-test skipped: rocSHMEM ${rocshmem_VERSION} lacks the host-stream APIs "
        "that this example requires (e.g. rocshmem_putmem_on_stream); rocSHMEM >= 3.6.0 is needed."
    )
    return()
endif()

# rocSHMEM's cmake config omits find_dependency(rocprofiler-register) despite
# roc::rocshmem carrying it as a transitive link dependency.  Ensure the
# imported target exists so the linker receives a library path rather than the
# raw target name when building against an installed rocSHMEM package.
if(NOT TARGET rocprofiler-register::rocprofiler-register)
    find_package(rocprofiler-register QUIET)
endif()

# MPI is optional at the source level — rocshmem_init() handles MPI setup
# internally — but linking MPI ensures correct symbol resolution when the
# demo is launched via mpirun.
find_package(MPI QUIET)

enable_language(HIP)

add_executable(rocshmem-test rocshmem.cpp)
set_source_files_properties(rocshmem.cpp PROPERTIES LANGUAGE HIP)

set_target_properties(
    rocshmem-test
    PROPERTIES
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

target_link_libraries(
    rocshmem-test
    PRIVATE
        roc::rocshmem
        hip::host
        $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
        $<TARGET_NAME_IF_EXISTS:rocprofiler-systems::rocprofiler-systems-compile-options>
)

if(ROCPROFSYS_INSTALL_EXAMPLES)
    install(
        TARGETS rocshmem-test
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
        OPTIONAL
    )
endif()
