# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

set(YAML_CPP_VERSION "0.9.0" CACHE STRING "yaml-cpp version")

find_package(yaml-cpp ${YAML_CPP_VERSION} QUIET)

if(yaml-cpp_FOUND)
    message(STATUS "Using system yaml-cpp (version ${yaml-cpp_VERSION})")
    get_target_property(
        _yaml_cpp_includes
        yaml-cpp::yaml-cpp
        INTERFACE_INCLUDE_DIRECTORIES
    )
    if(_yaml_cpp_includes)
        set_target_properties(
            yaml-cpp::yaml-cpp
            PROPERTIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_yaml_cpp_includes}"
        )
    endif()
else()
    message(
        STATUS
        "System yaml-cpp not found, fetching version ${YAML_CPP_VERSION}"
    )
    include(FetchContent)

    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-${YAML_CPP_VERSION}
        GIT_SHALLOW TRUE
    )

    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
    set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    set(_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)

    FetchContent_MakeAvailable(yaml-cpp)

    set(BUILD_SHARED_LIBS ${_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP})
    unset(_PROFILER_HUB_BUILD_SHARED_LIBS_BACKUP)

    if(TARGET yaml-cpp)
        set_target_properties(yaml-cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
        get_target_property(
            _yaml_cpp_includes
            yaml-cpp
            INTERFACE_INCLUDE_DIRECTORIES
        )
        if(_yaml_cpp_includes)
            set_target_properties(
                yaml-cpp
                PROPERTIES
                    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_yaml_cpp_includes}"
            )
        endif()
    endif()

    if(NOT TARGET yaml-cpp::yaml-cpp)
        add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
    endif()
endif()
