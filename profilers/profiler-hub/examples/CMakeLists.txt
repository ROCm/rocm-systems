# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(profiler-hub-examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Resolve the library target.
if(TARGET profiler-hub)
    # In-tree: the main build already defines the target.
    set(PROFILER_HUB_EXAMPLE_LIBRARY profiler-hub)
elseif(TARGET profiler-hub::profiler-hub)
    set(PROFILER_HUB_EXAMPLE_LIBRARY profiler-hub::profiler-hub)
else()
    # Standalone: try an installed/exported package first (only when the caller
    # actually pointed us at one, to avoid erroring on a half-exported build
    # tree config).
    if(DEFINED profiler-hub_DIR OR DEFINED CMAKE_PREFIX_PATH)
        find_package(profiler-hub QUIET)
    endif()

    if(TARGET profiler-hub::profiler-hub)
        set(PROFILER_HUB_EXAMPLE_LIBRARY profiler-hub::profiler-hub)
    else()
        # Build-tree fallback: import the prebuilt shared library directly.
        set(PROFILER_HUB_BUILD_DIR
            "${CMAKE_CURRENT_LIST_DIR}/../build"
            CACHE PATH
            "Existing profiler-hub build tree to link against"
        )

        find_library(
            PROFILER_HUB_PREBUILT_LIB
            NAMES profiler-hub
            HINTS "${PROFILER_HUB_BUILD_DIR}/lib"
            NO_DEFAULT_PATH
        )

        if(
            NOT PROFILER_HUB_PREBUILT_LIB
            OR NOT EXISTS "${PROFILER_HUB_BUILD_DIR}/include"
        )
            message(
                FATAL_ERROR
                "Could not locate profiler-hub. Either install it and pass "
                "-DCMAKE_PREFIX_PATH=<install>, or set "
                "-DPROFILER_HUB_BUILD_DIR=<existing build tree>. Looked in "
                "'${PROFILER_HUB_BUILD_DIR}'."
            )
        endif()

        add_library(profiler-hub::profiler-hub SHARED IMPORTED)
        set_target_properties(
            profiler-hub::profiler-hub
            PROPERTIES
                IMPORTED_LOCATION "${PROFILER_HUB_PREBUILT_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES
                    "${PROFILER_HUB_BUILD_DIR}/include"
        )
        set(PROFILER_HUB_EXAMPLE_LIBRARY profiler-hub::profiler-hub)

        message(
            STATUS
            "profiler-hub examples: using build-tree library "
            "${PROFILER_HUB_PREBUILT_LIB}"
        )
    endif()
endif()

# Whether to install the example programs. Defaults ON for a standalone build
# and OFF when built as part of the main project (matches systems' behavior).
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(PROFILER_HUB_INSTALL_EXAMPLES "Install rocprofiler-hub examples" ON)
else()
    option(PROFILER_HUB_INSTALL_EXAMPLES "Install rocprofiler-hub examples" OFF)
endif()

if(PROFILER_HUB_INSTALL_EXAMPLES)
    include(GNUInstallDirs)
endif()

# Where the built example executables are placed. They are kept under the
# profiler-hub build tree at <build>/bin/examples, regardless of where this
# project is configured from.
if(DEFINED PROFILER_HUB_BUILD_DIR)
    set(_phub_example_root "${PROFILER_HUB_BUILD_DIR}")
else()
    set(_phub_example_root "${CMAKE_BINARY_DIR}")
endif()
set(PROFILER_HUB_EXAMPLE_OUTPUT_DIR
    "${_phub_example_root}/bin/examples"
    CACHE PATH
    "Output directory for the built example executables"
)

add_subdirectory(schema_v3)
