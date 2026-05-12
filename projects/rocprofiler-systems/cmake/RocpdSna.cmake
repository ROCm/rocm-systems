# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc. All Rights Reserved.

include_guard(GLOBAL)

# ------------------------------------------------------------------------------
# RocpdSna integration for rocprofiler-systems.
#
# Resolution order:
#   1. find_package(rocpdsna) - picks up a system or externally-installed copy
#      (looks at CMAKE_PREFIX_PATH, rocpdsna_ROOT, etc.)
#   2. In-tree sibling at ${PROJECT_SOURCE_DIR}/../rocpdsna - the natural
#      monorepo layout for rocm-systems (projects/rocpdsna alongside
#      projects/rocprofiler-systems)
#   3. FetchContent from the rocm-systems repository - shallow clone with
#      sparse-checkout limited to projects/rocpdsna and the cmake helpers
#      it needs. Last resort, intended for stand-alone consumers that do
#      not already have the monorepo on disk.
#
# All three paths produce the same consumer-facing target:
#     rocprofiler-systems::rocprofiler-systems-rocpdsna
# ------------------------------------------------------------------------------

option(ROCPROFSYS_USE_SNA "Enable rocpdsna integration" ON)

if(NOT ROCPROFSYS_USE_SNA)
    message(STATUS "[rocpdsna] disabled (ROCPROFSYS_USE_SNA=OFF)")
    add_library(rocprofiler-systems-rocpdsna INTERFACE)
    add_library(
        rocprofiler-systems::rocprofiler-systems-rocpdsna
        ALIAS rocprofiler-systems-rocpdsna
    )
    return()
endif()

option(ROCPROFSYS_ROCPDSNA_LINK_STATIC "Link rocpdsna statically" OFF)
option(ROCPROFSYS_ROCPDSNA_ENABLE_LOGGING "Enable rocpdsna logging" OFF)

# Pin a SHA for the FetchContent fallback so reproducible builds do not drift
# with develop. Update intentionally when bumping the rocpdsna ABI.
set(ROCPROFSYS_ROCPDSNA_GIT_REPOSITORY
    "https://github.com/ROCm/rocm-systems.git"
    CACHE STRING
    "Git repository for FetchContent fallback when rocpdsna is not found locally"
)
set(ROCPROFSYS_ROCPDSNA_GIT_TAG
    "sna-develop"
    CACHE STRING
    "Git ref to fetch for the FetchContent fallback (prefer a SHA for reproducibility)"
)

# ------------------------------------------------------------------------------
# Step 1: find_package
# ------------------------------------------------------------------------------

find_package(rocpdsna QUIET)
set(_ROCPDSNA_RESOLVED FALSE)

if(rocpdsna_FOUND)
    message(STATUS "[rocpdsna] resolved via find_package: ${rocpdsna_DIR}")
    set(_ROCPDSNA_IS_EXTERNAL TRUE)
    set(_ROCPDSNA_RESOLVED TRUE)
endif()

# ------------------------------------------------------------------------------
# Step 2: in-tree sibling (monorepo layout)
# ------------------------------------------------------------------------------

if(NOT _ROCPDSNA_RESOLVED)
    set(_ROCPDSNA_SIBLING_DIR "${PROJECT_SOURCE_DIR}/../rocpdsna")
    if(EXISTS "${_ROCPDSNA_SIBLING_DIR}/CMakeLists.txt")
        message(
            STATUS
            "[rocpdsna] resolved via in-tree sibling: ${_ROCPDSNA_SIBLING_DIR}"
        )
        set(ROCPDSNA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(ROCPDSNA_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
        set(ROCPDSNA_ENABLE_COVERAGE OFF CACHE BOOL "" FORCE)
        set(ROCPDSNA_ENABLE_LOGGING
            ${ROCPROFSYS_ROCPDSNA_ENABLE_LOGGING}
            CACHE BOOL
            ""
            FORCE
        )
        add_subdirectory(
            ${_ROCPDSNA_SIBLING_DIR}
            ${PROJECT_BINARY_DIR}/external/rocpdsna
            EXCLUDE_FROM_ALL
        )
        set(_ROCPDSNA_IS_EXTERNAL FALSE)
        set(_ROCPDSNA_RESOLVED TRUE)
    endif()
endif()

# ------------------------------------------------------------------------------
# Step 3: FetchContent fallback (sparse checkout of rocm-systems)
# ------------------------------------------------------------------------------

if(NOT _ROCPDSNA_RESOLVED)
    message(
        STATUS
        "[rocpdsna] not found locally; fetching via FetchContent from "
        "${ROCPROFSYS_ROCPDSNA_GIT_REPOSITORY} (${ROCPROFSYS_ROCPDSNA_GIT_TAG})"
    )

    include(FetchContent)

    # Sparse checkout pattern: keep CMake configure-time cost bounded by only
    # materializing the rocpdsna subtree (and the top-level CMake helpers it
    # references through its own includes). FetchContent does not expose
    # sparse-checkout natively, so we drive it through git via a custom
    # GIT_CONFIG flag and a post-clone configure step.
    # GIT_CONFIG values bundled in a list so the formatter cannot reflow
    # SOURCE_DIR / BINARY_DIR / SUBBUILD_DIR into the GIT_CONFIG argument list.
    set(_rocpdsna_git_config
        "core.sparseCheckout=true"
        "remote.origin.partialclonefilter=blob:none"
    )
    FetchContent_Declare(
        rocpdsna_src
        SOURCE_DIR
        ${PROJECT_BINARY_DIR}/external/rocpdsna/src
        BINARY_DIR
        ${PROJECT_BINARY_DIR}/external/rocpdsna/build
        SUBBUILD_DIR
        ${PROJECT_BINARY_DIR}/external/rocpdsna/subbuild
        GIT_REPOSITORY ${ROCPROFSYS_ROCPDSNA_GIT_REPOSITORY}
        GIT_TAG ${ROCPROFSYS_ROCPDSNA_GIT_TAG}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        GIT_CONFIG ${_rocpdsna_git_config}
    )

    FetchContent_GetProperties(rocpdsna_src)
    if(NOT rocpdsna_src_POPULATED)
        FetchContent_Populate(rocpdsna_src)

        # Materialize only the rocpdsna subtree
        set(_sparse_file "${rocpdsna_src_SOURCE_DIR}/.git/info/sparse-checkout")
        file(WRITE "${_sparse_file}" "projects/rocpdsna/\n")
        execute_process(
            COMMAND git -C "${rocpdsna_src_SOURCE_DIR}" read-tree -mu HEAD
            RESULT_VARIABLE _sparse_rc
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(NOT _sparse_rc EQUAL 0)
            message(
                WARNING
                "[rocpdsna] sparse-checkout post-step failed (rc=${_sparse_rc}); "
                "the full tree will be present but build still works"
            )
        endif()

        set(ROCPDSNA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(ROCPDSNA_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
        set(ROCPDSNA_ENABLE_COVERAGE OFF CACHE BOOL "" FORCE)
        set(ROCPDSNA_ENABLE_LOGGING
            ${ROCPROFSYS_ROCPDSNA_ENABLE_LOGGING}
            CACHE BOOL
            ""
            FORCE
        )

        add_subdirectory(
            ${rocpdsna_src_SOURCE_DIR}/projects/rocpdsna
            ${PROJECT_BINARY_DIR}/external/rocpdsna/build
            EXCLUDE_FROM_ALL
        )
    endif()

    set(_ROCPDSNA_IS_EXTERNAL FALSE)
    set(_ROCPDSNA_RESOLVED TRUE)
endif()

if(NOT _ROCPDSNA_RESOLVED)
    message(FATAL_ERROR "[rocpdsna] could not resolve via any path")
endif()

# ------------------------------------------------------------------------------
# Pick concrete target name (static vs. shared, external vs. in-tree)
# ------------------------------------------------------------------------------

if(ROCPROFSYS_ROCPDSNA_LINK_STATIC)
    set(_ROCPDSNA_SUFFIX "-static")
    message(STATUS "[rocpdsna] linking statically")
else()
    set(_ROCPDSNA_SUFFIX "")
endif()

if(_ROCPDSNA_IS_EXTERNAL)
    set(_ROCPDSNA_TARGET rocpdsna::rocpdsna${_ROCPDSNA_SUFFIX})
else()
    set(_ROCPDSNA_TARGET rocpdsna${_ROCPDSNA_SUFFIX})
endif()

# ------------------------------------------------------------------------------
# Public consumer target
# ------------------------------------------------------------------------------

add_library(rocprofiler-systems-rocpdsna INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-rocpdsna
    ALIAS rocprofiler-systems-rocpdsna
)
target_link_libraries(rocprofiler-systems-rocpdsna INTERFACE ${_ROCPDSNA_TARGET})

# For non-installed sources, propagate the include directories explicitly
# and mark them SYSTEM so rocpdsna's own warning policy does not trip
# rocprofiler-systems' -Werror.
if(NOT _ROCPDSNA_IS_EXTERNAL)
    target_include_directories(
        rocprofiler-systems-rocpdsna
        SYSTEM
        INTERFACE $<TARGET_PROPERTY:rocpdsna,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()
