include_guard(GLOBAL)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------

option(
    ROCPROFSYS_USE_EXTERNAL_ROCSTORAGE
    "Use externally installed rocstorage library instead of building from source"
    OFF
)

set(ROCPROFSYS_ROCSTORAGE_GIT_REPOSITORY
    ""
    CACHE STRING
    "Git repository URL for rocstorage (leave empty to use local path)"
)

set(ROCPROFSYS_ROCSTORAGE_GIT_TAG "develop" CACHE STRING "Git tag/branch for rocstorage")

set(ROCPROFSYS_ROCSTORAGE_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/../rocstorage"
    CACHE PATH
    "Local path to rocstorage source directory"
)

option(ROCPROFSYS_ROCSTORAGE_ENABLE_LOGGING "Enable rocstorage logging" OFF)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

if(ROCPROFSYS_USE_EXTERNAL_ROCSTORAGE)
    find_package(rocstorage REQUIRED)
    message(STATUS "[rocstorage] Using external installation: ${rocstorage_DIR}")

    set(_ROCSTORAGE_TARGET rocstorage::rocstorage)
    set(_ROCSTORAGE_IS_EXTERNAL TRUE)
elseif(ROCPROFSYS_ROCSTORAGE_GIT_REPOSITORY)
    include(FetchContent)

    set(ROCSTORAGE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ROCSTORAGE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(ROCSTORAGE_ENABLE_LOGGING
        ${ROCPROFSYS_ROCSTORAGE_ENABLE_LOGGING}
        CACHE BOOL
        ""
        FORCE
    )

    message(
        STATUS
        "[rocstorage] Fetching from Git: ${ROCPROFSYS_ROCSTORAGE_GIT_REPOSITORY} (${ROCPROFSYS_ROCSTORAGE_GIT_TAG})"
    )
    FetchContent_Declare(
        rocstorage
        GIT_REPOSITORY ${ROCPROFSYS_ROCSTORAGE_GIT_REPOSITORY}
        GIT_TAG ${ROCPROFSYS_ROCSTORAGE_GIT_TAG}
        SOURCE_DIR
        ${PROJECT_BINARY_DIR}/external/rocstorage/src
        BINARY_DIR
        ${PROJECT_BINARY_DIR}/external/rocstorage/build
        SUBBUILD_DIR
        ${PROJECT_BINARY_DIR}/external/rocstorage/subbuild
    )
    FetchContent_MakeAvailable(rocstorage)

    set(_ROCSTORAGE_TARGET rocstorage)
    set(_ROCSTORAGE_IS_EXTERNAL FALSE)
else()
    include(FetchContent)

    set(ROCSTORAGE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ROCSTORAGE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(ROCSTORAGE_ENABLE_LOGGING
        ${ROCPROFSYS_ROCSTORAGE_ENABLE_LOGGING}
        CACHE BOOL
        ""
        FORCE
    )

    message(STATUS "[rocstorage] Using local source: ${ROCPROFSYS_ROCSTORAGE_SOURCE_DIR}")
    FetchContent_Declare(
        rocstorage
        SOURCE_DIR
        ${ROCPROFSYS_ROCSTORAGE_SOURCE_DIR}
        BINARY_DIR
        ${PROJECT_BINARY_DIR}/external/rocstorage/build
        SUBBUILD_DIR
        ${PROJECT_BINARY_DIR}/external/rocstorage/subbuild
    )
    FetchContent_MakeAvailable(rocstorage)

    set(_ROCSTORAGE_TARGET rocstorage)
    set(_ROCSTORAGE_IS_EXTERNAL FALSE)
endif()

# ------------------------------------------------------------------------------
# Interface target
# ------------------------------------------------------------------------------

add_library(rocprofiler-systems-rocstorage INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-rocstorage
    ALIAS rocprofiler-systems-rocstorage
)
target_link_libraries(rocprofiler-systems-rocstorage INTERFACE ${_ROCSTORAGE_TARGET})

if(NOT _ROCSTORAGE_IS_EXTERNAL)
    target_include_directories(
        rocprofiler-systems-rocstorage
        SYSTEM
        INTERFACE $<TARGET_PROPERTY:rocstorage,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()
