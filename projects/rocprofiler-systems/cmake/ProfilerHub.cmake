# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------

set(ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY
    "https://github.com/ROCm/rocm-systems.git"
    CACHE STRING
    "Git repository URL for profiler-hub fallback sparse checkout"
)

set(ROCPROFSYS_PROFILER_HUB_GIT_TAG
    "develop"
    CACHE STRING
    "Git tag/branch for profiler-hub fallback sparse checkout"
)

set(ROCPROFSYS_PROFILER_HUB_GIT_SUBDIR
    "profilers/profiler-hub"
    CACHE STRING
    "Subdirectory inside the repository that contains profiler-hub"
)

set(ROCPROFSYS_PROFILER_HUB_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/../../profilers/profiler-hub"
    CACHE PATH
    "Optional local in-tree path to profiler-hub source (used if it exists)"
)

option(ROCPROFSYS_PROFILER_HUB_ENABLE_LOGGING "Enable profiler-hub logging" OFF)
option(ROCPROFSYS_PROFILER_HUB_LINK_STATIC "Link profiler-hub statically" OFF)

# ------------------------------------------------------------------------------
# Resolution order:
#   1. find_package(profiler-hub) - use installed package if present
#   2. local in-tree source at ROCPROFSYS_PROFILER_HUB_SOURCE_DIR (CI / monorepo)
#   3. fallback: sparse checkout of ROCPROFSYS_PROFILER_HUB_GIT_SUBDIR from
#      ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY at ROCPROFSYS_PROFILER_HUB_GIT_TAG
# ------------------------------------------------------------------------------

set(_PROFILER_HUB_IS_EXTERNAL FALSE)
set(_PROFILER_HUB_SOURCE_DIR "")

find_package(profiler-hub QUIET CONFIG)

if(profiler-hub_FOUND)
    message(STATUS "[profiler-hub] Using installed package: ${profiler-hub_DIR}")
    set(_PROFILER_HUB_IS_EXTERNAL TRUE)
elseif(EXISTS "${ROCPROFSYS_PROFILER_HUB_SOURCE_DIR}/CMakeLists.txt")
    message(
        STATUS
        "[profiler-hub] Using in-tree source: ${ROCPROFSYS_PROFILER_HUB_SOURCE_DIR}"
    )
    set(_PROFILER_HUB_SOURCE_DIR "${ROCPROFSYS_PROFILER_HUB_SOURCE_DIR}")
else()
    message(
        STATUS
        "[profiler-hub] find_package failed and no in-tree source; falling back to "
        "sparse checkout of ${ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY} "
        "(${ROCPROFSYS_PROFILER_HUB_GIT_TAG})"
    )

    find_package(Git REQUIRED)

    set(_PROFILER_HUB_ROOT "${PROJECT_BINARY_DIR}/external/profiler-hub")
    set(_PROFILER_HUB_CHECKOUT "${_PROFILER_HUB_ROOT}/src")
    set(_PROFILER_HUB_SOURCE_DIR
        "${_PROFILER_HUB_CHECKOUT}/${ROCPROFSYS_PROFILER_HUB_GIT_SUBDIR}"
    )
    set(_PROFILER_HUB_STAMP "${_PROFILER_HUB_ROOT}/.checkout.stamp")

    set(_PROFILER_HUB_NEEDS_CHECKOUT TRUE)
    if(EXISTS "${_PROFILER_HUB_STAMP}")
        file(READ "${_PROFILER_HUB_STAMP}" _stamp_contents)
        if(
            _stamp_contents
                STREQUAL
                "${ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY}@${ROCPROFSYS_PROFILER_HUB_GIT_TAG}:${ROCPROFSYS_PROFILER_HUB_GIT_SUBDIR}"
        )
            set(_PROFILER_HUB_NEEDS_CHECKOUT FALSE)
        endif()
    endif()

    if(_PROFILER_HUB_NEEDS_CHECKOUT)
        message(
            STATUS
            "[profiler-hub] Sparse-checking out ${ROCPROFSYS_PROFILER_HUB_GIT_SUBDIR} into ${_PROFILER_HUB_CHECKOUT}"
        )

        if(EXISTS "${_PROFILER_HUB_CHECKOUT}")
            file(REMOVE_RECURSE "${_PROFILER_HUB_CHECKOUT}")
        endif()
        file(MAKE_DIRECTORY "${_PROFILER_HUB_CHECKOUT}")

        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} clone --filter=blob:none --no-checkout --depth 1
                --branch ${ROCPROFSYS_PROFILER_HUB_GIT_TAG}
                ${ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY} ${_PROFILER_HUB_CHECKOUT}
            RESULT_VARIABLE _git_result
            OUTPUT_VARIABLE _git_output
            ERROR_VARIABLE _git_error
        )
        if(NOT _git_result EQUAL 0)
            message(
                FATAL_ERROR
                "[profiler-hub] git clone failed (${_git_result}): ${_git_error}\n${_git_output}"
            )
        endif()

        execute_process(
            COMMAND ${GIT_EXECUTABLE} sparse-checkout init --cone
            WORKING_DIRECTORY ${_PROFILER_HUB_CHECKOUT}
            RESULT_VARIABLE _git_result
            ERROR_VARIABLE _git_error
        )
        if(NOT _git_result EQUAL 0)
            message(
                FATAL_ERROR
                "[profiler-hub] sparse-checkout init failed: ${_git_error}"
            )
        endif()

        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} sparse-checkout set
                ${ROCPROFSYS_PROFILER_HUB_GIT_SUBDIR}
            WORKING_DIRECTORY ${_PROFILER_HUB_CHECKOUT}
            RESULT_VARIABLE _git_result
            ERROR_VARIABLE _git_error
        )
        if(NOT _git_result EQUAL 0)
            message(
                FATAL_ERROR
                "[profiler-hub] sparse-checkout set failed: ${_git_error}"
            )
        endif()

        execute_process(
            COMMAND ${GIT_EXECUTABLE} checkout ${ROCPROFSYS_PROFILER_HUB_GIT_TAG}
            WORKING_DIRECTORY ${_PROFILER_HUB_CHECKOUT}
            RESULT_VARIABLE _git_result
            ERROR_VARIABLE _git_error
        )
        if(NOT _git_result EQUAL 0)
            message(FATAL_ERROR "[profiler-hub] git checkout failed: ${_git_error}")
        endif()

        if(NOT EXISTS "${_PROFILER_HUB_SOURCE_DIR}/CMakeLists.txt")
            message(
                FATAL_ERROR
                "[profiler-hub] Sparse checkout completed but CMakeLists.txt missing at "
                "${_PROFILER_HUB_SOURCE_DIR}"
            )
        endif()

        file(
            WRITE "${_PROFILER_HUB_STAMP}"
            "${ROCPROFSYS_PROFILER_HUB_GIT_REPOSITORY}@${ROCPROFSYS_PROFILER_HUB_GIT_TAG}:${ROCPROFSYS_PROFILER_HUB_GIT_SUBDIR}"
        )
    else()
        message(
            STATUS
            "[profiler-hub] Reusing existing sparse checkout at ${_PROFILER_HUB_SOURCE_DIR}"
        )
    endif()
endif()

if(NOT _PROFILER_HUB_IS_EXTERNAL)
    set(PROFILER_HUB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PROFILER_HUB_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(PROFILER_HUB_ENABLE_LOGGING
        ${ROCPROFSYS_PROFILER_HUB_ENABLE_LOGGING}
        CACHE BOOL
        ""
        FORCE
    )

    add_subdirectory(
        ${_PROFILER_HUB_SOURCE_DIR}
        ${PROJECT_BINARY_DIR}/external/profiler-hub/build
    )
endif()

if(ROCPROFSYS_PROFILER_HUB_LINK_STATIC)
    set(_PROFILER_HUB_SUFFIX "-static")
    message(STATUS "[profiler-hub] Linking statically")
else()
    set(_PROFILER_HUB_SUFFIX "")
endif()

if(_PROFILER_HUB_IS_EXTERNAL)
    set(_PROFILER_HUB_TARGET profiler-hub::profiler-hub${_PROFILER_HUB_SUFFIX})
else()
    set(_PROFILER_HUB_TARGET profiler-hub${_PROFILER_HUB_SUFFIX})
endif()

# ------------------------------------------------------------------------------
# Interface target
# ------------------------------------------------------------------------------

add_library(rocprofiler-systems-profiler-hub INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-profiler-hub
    ALIAS rocprofiler-systems-profiler-hub
)
target_link_libraries(rocprofiler-systems-profiler-hub INTERFACE ${_PROFILER_HUB_TARGET})

if(NOT _PROFILER_HUB_IS_EXTERNAL)
    target_include_directories(
        rocprofiler-systems-profiler-hub
        SYSTEM
        INTERFACE $<TARGET_PROPERTY:profiler-hub,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()
