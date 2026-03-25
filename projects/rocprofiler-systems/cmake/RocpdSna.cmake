include_guard(GLOBAL)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------

option(
    ROCPROFSYS_USE_EXTERNAL_ROCPDSNA
    "Use externally installed rocpdsna library instead of building from source"
    OFF
)

set(ROCPROFSYS_ROCPDSNA_GIT_REPOSITORY
    ""
    CACHE STRING
    "Git repository URL for rocpdsna (leave empty to use local path)"
)

set(ROCPROFSYS_ROCPDSNA_GIT_TAG "develop" CACHE STRING "Git tag/branch for rocpdsna")

set(ROCPROFSYS_ROCPDSNA_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/../rocpdsna"
    CACHE PATH
    "Local path to rocpdsna source directory"
)

option(ROCPROFSYS_ROCPDSNA_ENABLE_LOGGING "Enable rocpdsna logging" OFF)
option(ROCPROFSYS_ROCPDSNA_LINK_STATIC "Link rocpdsna statically" OFF)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

if(ROCPROFSYS_USE_EXTERNAL_ROCPDSNA)
    find_package(rocpdsna REQUIRED)
    message(STATUS "[rocpdsna] Using external installation: ${rocpdsna_DIR}")

    set(_ROCPDSNA_IS_EXTERNAL TRUE)
else()
    include(FetchContent)

    set(ROCPDSNA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ROCPDSNA_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(ROCPDSNA_ENABLE_LOGGING ${ROCPROFSYS_ROCPDSNA_ENABLE_LOGGING} CACHE BOOL "" FORCE)

    if(ROCPROFSYS_ROCPDSNA_GIT_REPOSITORY)
        message(
            STATUS
            "[rocpdsna] Fetching from Git: ${ROCPROFSYS_ROCPDSNA_GIT_REPOSITORY} (${ROCPROFSYS_ROCPDSNA_GIT_TAG})"
        )
        set(_ROCPDSNA_SOURCE_ARGS
            GIT_REPOSITORY
            ${ROCPROFSYS_ROCPDSNA_GIT_REPOSITORY}
            GIT_TAG
            ${ROCPROFSYS_ROCPDSNA_GIT_TAG}
            SOURCE_DIR
            ${PROJECT_BINARY_DIR}/external/rocpdsna/src
        )
    else()
        message(STATUS "[rocpdsna] Using local source: ${ROCPROFSYS_ROCPDSNA_SOURCE_DIR}")
        set(_ROCPDSNA_SOURCE_ARGS SOURCE_DIR ${ROCPROFSYS_ROCPDSNA_SOURCE_DIR})
    endif()

    FetchContent_Declare(
        rocpdsna
        ${_ROCPDSNA_SOURCE_ARGS}
        BINARY_DIR
        ${PROJECT_BINARY_DIR}/external/rocpdsna/build
        SUBBUILD_DIR
        ${PROJECT_BINARY_DIR}/external/rocpdsna/subbuild
    )
    FetchContent_MakeAvailable(rocpdsna)

    set(_ROCPDSNA_IS_EXTERNAL FALSE)
endif()

if(ROCPROFSYS_ROCPDSNA_LINK_STATIC)
    set(_ROCPDSNA_SUFFIX "-static")
    message(STATUS "[rocpdsna] Linking statically")
else()
    set(_ROCPDSNA_SUFFIX "")
endif()

if(_ROCPDSNA_IS_EXTERNAL)
    set(_ROCPDSNA_TARGET rocpdsna::rocpdsna${_ROCPDSNA_SUFFIX})
else()
    set(_ROCPDSNA_TARGET rocpdsna${_ROCPDSNA_SUFFIX})
endif()

# ------------------------------------------------------------------------------
# Interface target
# ------------------------------------------------------------------------------

add_library(rocprofiler-systems-rocpdsna INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-rocpdsna
    ALIAS rocprofiler-systems-rocpdsna
)
target_link_libraries(rocprofiler-systems-rocpdsna INTERFACE ${_ROCPDSNA_TARGET})

if(NOT _ROCPDSNA_IS_EXTERNAL)
    target_include_directories(
        rocprofiler-systems-rocpdsna
        SYSTEM
        INTERFACE $<TARGET_PROPERTY:rocpdsna,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()
