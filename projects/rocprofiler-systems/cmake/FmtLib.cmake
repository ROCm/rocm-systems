# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

if(ROCPROFSYS_BUILD_FMT)
    message(STATUS "Building fmt from source!")

    include(FetchContent)

    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/fmt
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/fmtlib/fmt.git
        TEST_FILE CMakeLists.txt
        REPO_BRANCH "12.2.0"
    )

    FetchContent_Declare(fmt SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/fmt)

    set(FMT_DOC OFF CACHE BOOL "" FORCE)
    set(FMT_TEST OFF CACHE BOOL "" FORCE)
    set(FMT_INSTALL OFF CACHE BOOL "" FORCE)

    # fmt workaround for building static library
    set(_ROCPROFSYS_BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)

    FetchContent_MakeAvailable(fmt)

    set(BUILD_SHARED_LIBS ${_ROCPROFSYS_BUILD_SHARED_LIBS_BACKUP})
    unset(_ROCPROFSYS_BUILD_SHARED_LIBS_BACKUP)

    # Mark fmt include directories as SYSTEM so bundled-third-party warnings
    # (e.g. GCC 10 -Wstringop-overflow false positive in fmt::v12::detail::write)
    # do not break our -Werror builds.
    get_target_property(_fmt_include_dirs fmt INTERFACE_INCLUDE_DIRECTORIES)
    if(_fmt_include_dirs)
        set_target_properties(
            fmt
            PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_fmt_include_dirs}"
        )
    endif()

    target_link_libraries(rocprofiler-systems-fmt INTERFACE fmt::fmt)
else()
    message(STATUS "Using system fmt library")
    find_package(fmt REQUIRED)

    # fmt::fmt may be an ALIAS to the real imported target (e.g. Debian's
    # fmtConfig.cmake), and set_target_properties() rejects ALIAS targets.
    get_target_property(_fmt_aliased_target fmt::fmt ALIASED_TARGET)
    if(_fmt_aliased_target)
        set(_fmt_target ${_fmt_aliased_target})
    else()
        set(_fmt_target fmt::fmt)
    endif()

    get_target_property(_fmt_include_dirs ${_fmt_target} INTERFACE_INCLUDE_DIRECTORIES)
    if(_fmt_include_dirs)
        set_target_properties(
            ${_fmt_target}
            PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_fmt_include_dirs}"
        )
    endif()

    target_link_libraries(rocprofiler-systems-fmt INTERFACE fmt::fmt)
endif()
