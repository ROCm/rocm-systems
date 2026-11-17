# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# Same minimum profiler-hub requires. Build the submodule only when that
# version is not already installed.
find_package(spdlog 1.15.3 QUIET)

if(NOT spdlog_FOUND AND ROCPROFSYS_BUILD_SPDLOG)
    message(STATUS "Building spdlog from source!")

    include(FetchContent)

    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/spdlog
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/gabime/spdlog.git
        TEST_FILE CMakeLists.txt
        REPO_BRANCH "v1.17.0"
    )

    FetchContent_Declare(spdlog SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/spdlog)

    set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
    set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)

    # Spdlog workaround for building static library
    set(_ROCPROFSYS_BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)

    FetchContent_MakeAvailable(spdlog)

    set(BUILD_SHARED_LIBS ${_ROCPROFSYS_BUILD_SHARED_LIBS_BACKUP})
    unset(_ROCPROFSYS_BUILD_SHARED_LIBS_BACKUP)

    # Mark spdlog include directories as SYSTEM so bundled-third-party warnings
    # (e.g. GCC 10 -Wstringop-overflow false positive in fmt::v12::detail::write)
    # do not break our -Werror builds. fmt's own include directories are marked
    # SYSTEM in FmtLib.cmake.
    get_target_property(_spdlog_include_dirs spdlog INTERFACE_INCLUDE_DIRECTORIES)
    if(_spdlog_include_dirs)
        set_target_properties(
            spdlog
            PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_spdlog_include_dirs}"
        )
    endif()

    target_link_libraries(
        rocprofiler-systems-spdlog
        INTERFACE spdlog::spdlog rocprofiler-systems::rocprofiler-systems-fmt
    )
    return()
endif()

if(NOT spdlog_FOUND)
    find_package(spdlog REQUIRED)
endif()

message(STATUS "Using system spdlog library")

# spdlog::spdlog may be an ALIAS to the real imported target (e.g. Debian's
# spdlogConfig.cmake), and set_target_properties() rejects ALIAS targets.
get_target_property(_spdlog_aliased_target spdlog::spdlog ALIASED_TARGET)
if(_spdlog_aliased_target)
    set(_spdlog_target ${_spdlog_aliased_target})
else()
    set(_spdlog_target spdlog::spdlog)
endif()

get_target_property(_spdlog_include_dirs ${_spdlog_target} INTERFACE_INCLUDE_DIRECTORIES)
if(_spdlog_include_dirs)
    set_target_properties(
        ${_spdlog_target}
        PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_spdlog_include_dirs}"
    )
endif()
target_link_libraries(
    rocprofiler-systems-spdlog
    INTERFACE spdlog::spdlog rocprofiler-systems::rocprofiler-systems-fmt
)
