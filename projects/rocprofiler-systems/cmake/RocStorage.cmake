include_guard(GLOBAL)

include(FetchContent)

set(ROCPROFSYS_ROCSTORAGE_GIT_REPOSITORY
    ""
    CACHE STRING
    "Git repository URL for rocstorage (leave empty to use local path)"
)

set(ROCPROFSYS_ROCSTORAGE_GIT_TAG "main" CACHE STRING "Git tag/branch for rocstorage")

set(ROCPROFSYS_ROCSTORAGE_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/../rocstorage"
    CACHE PATH
    "Local path to rocstorage source directory"
)

set(ROCSTORAGE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ROCSTORAGE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

option(ROCPROFSYS_ROCSTORAGE_ENABLE_LOGGING "Enable rocstorage logging" OFF)
set(ROCSTORAGE_ENABLE_LOGGING ${ROCPROFSYS_ROCSTORAGE_ENABLE_LOGGING} CACHE BOOL "" FORCE)

# TODO: check if this is needed at all
if(ROCPROFSYS_ROCSTORAGE_GIT_REPOSITORY)
    message(STATUS "Fetching rocstorage from ${ROCPROFSYS_ROCSTORAGE_GIT_REPOSITORY}")
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
else()
    message(STATUS "Using local rocstorage from ${ROCPROFSYS_ROCSTORAGE_SOURCE_DIR}")
    FetchContent_Declare(
        rocstorage
        SOURCE_DIR
        ${ROCPROFSYS_ROCSTORAGE_SOURCE_DIR}
        BINARY_DIR
        ${PROJECT_BINARY_DIR}/external/rocstorage/build
        SUBBUILD_DIR
        ${PROJECT_BINARY_DIR}/external/rocstorage/subbuild
    )
endif()

FetchContent_MakeAvailable(rocstorage)

if(NOT TARGET rocprofiler-systems-rocstorage)
    add_library(rocprofiler-systems-rocstorage INTERFACE)
    add_library(
        rocprofiler-systems::rocprofiler-systems-rocstorage
        ALIAS rocprofiler-systems-rocstorage
    )

    target_link_libraries(rocprofiler-systems-rocstorage INTERFACE rocstorage)

    target_include_directories(
        rocprofiler-systems-rocstorage
        SYSTEM
        INTERFACE $<TARGET_PROPERTY:rocstorage,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()
