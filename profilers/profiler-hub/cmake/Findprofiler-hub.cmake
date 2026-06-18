# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#[=======================================================================[.rst:
Findprofiler-hub
--------------

Find the profiler-hub library.

Imported Targets
^^^^^^^^^^^^^^^^

``profiler-hub::profiler-hub``
  The profiler-hub library, if found.

Result Variables
^^^^^^^^^^^^^^^^

``profiler-hub_FOUND``
  True if profiler-hub was found.
``profiler-hub_INCLUDE_DIRS``
  Include directories for profiler-hub.
``profiler-hub_LIBRARIES``
  Libraries to link against.

#]=======================================================================]

# If target already exists (from add_subdirectory), we're done
if(TARGET profiler-hub)
    set(profiler-hub_FOUND TRUE)

    # Create namespaced aliases if they don't exist
    if(NOT TARGET profiler-hub::profiler-hub)
        add_library(profiler-hub::profiler-hub ALIAS profiler-hub)
    endif()

    if(
        TARGET profiler-hub-static
        AND NOT TARGET profiler-hub::profiler-hub-static
    )
        add_library(profiler-hub::profiler-hub-static ALIAS profiler-hub-static)
    endif()

    # Get properties from existing target
    get_target_property(
        profiler-hub_INCLUDE_DIRS
        profiler-hub
        INTERFACE_INCLUDE_DIRECTORIES
    )
    set(profiler-hub_LIBRARIES profiler-hub)

    return()
endif()

# If namespaced target exists (from previous find_package), we're done
if(TARGET profiler-hub::profiler-hub)
    set(profiler-hub_FOUND TRUE)
    return()
endif()

# Search for installed library
find_path(
    profiler-hub_INCLUDE_DIR
    NAMES profiler-hub/storage.hpp
    HINTS
        ${profiler-hub_ROOT}
        $ENV{profiler-hub_ROOT}
        ${CMAKE_BINARY_DIR}/include
    PATH_SUFFIXES include
)

find_library(
    profiler-hub_LIBRARY
    NAMES profiler-hub
    HINTS ${profiler-hub_ROOT} $ENV{profiler-hub_ROOT} ${CMAKE_BINARY_DIR}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    profiler-hub
    REQUIRED_VARS profiler-hub_LIBRARY profiler-hub_INCLUDE_DIR
)

if(profiler-hub_FOUND)
    set(profiler-hub_INCLUDE_DIRS ${profiler-hub_INCLUDE_DIR})
    set(profiler-hub_LIBRARIES ${profiler-hub_LIBRARY})

    if(NOT TARGET profiler-hub::profiler-hub)
        add_library(profiler-hub::profiler-hub SHARED IMPORTED)
        set_target_properties(
            profiler-hub::profiler-hub
            PROPERTIES
                IMPORTED_LOCATION "${profiler-hub_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${profiler-hub_INCLUDE_DIR}"
        )
    endif()

    # Also look for the static library
    find_library(
        profiler-hub_STATIC_LIBRARY
        NAMES libprofiler-hub.a profiler-hub
        HINTS ${profiler-hub_ROOT} $ENV{profiler-hub_ROOT} ${CMAKE_BINARY_DIR}
        PATH_SUFFIXES lib lib64
    )

    if(
        profiler-hub_STATIC_LIBRARY
        AND NOT TARGET profiler-hub::profiler-hub-static
    )
        add_library(profiler-hub::profiler-hub-static STATIC IMPORTED)
        set_target_properties(
            profiler-hub::profiler-hub-static
            PROPERTIES
                IMPORTED_LOCATION "${profiler-hub_STATIC_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${profiler-hub_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(
    profiler-hub_INCLUDE_DIR
    profiler-hub_LIBRARY
    profiler-hub_STATIC_LIBRARY
)
