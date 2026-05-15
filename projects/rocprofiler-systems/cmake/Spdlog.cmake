include_guard(GLOBAL)

if(ROCPROFSYS_BUILD_SPDLOG)
    message(STATUS "Building spdlog from source!")

    include(FetchContent)

    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/spdlog
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/gabime/spdlog.git
        TEST_FILE CMakeLists.txt
        REPO_BRANCH "v1.16.0"
    )

    FetchContent_Declare(spdlog SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/spdlog)

    set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
    set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)

    # Spdlog workaround for building static library
    set(_ROCPROFSYS_BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)

    FetchContent_MakeAvailable(spdlog)

    set(BUILD_SHARED_LIBS ${_ROCPROFSYS_BUILD_SHARED_LIBS_BACKUP})
    unset(_ROCPROFSYS_BUILD_SHARED_LIBS_BACKUP)

    target_link_libraries(rocprofiler-systems-spdlog INTERFACE spdlog::spdlog)
else()
    message(STATUS "Using system spdlog library")
    find_package(spdlog REQUIRED)
    target_link_libraries(rocprofiler-systems-spdlog INTERFACE spdlog::spdlog)
endif()
