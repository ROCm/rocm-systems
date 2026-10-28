# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# external/cpu-snap is a copy of a repository that is no longer maintained. It still builds
# with its own CMake; this file goes away when it is ported to first-party code.
message(STATUS "Building cpu-snap from external/cpu-snap")

# Give cpu-snap the libdw rocprof-sys already uses, so the process has one libdw. cpu-snap
# skips its own lookup when LibDw::LibDw exists.
if(NOT TARGET LibDW::LibDW)
    message(FATAL_ERROR "cpu-snap needs rocprofiler-systems' libdw (LibDW::LibDW)")
endif()
add_library(LibDw::LibDw INTERFACE IMPORTED)
target_link_libraries(LibDw::LibDw INTERFACE LibDW::LibDW)

set(CPU_SNAP_BUILD_SAMPLING ON CACHE BOOL "" FORCE)
set(CPU_SNAP_BUILD_PROCESSING OFF CACHE BOOL "" FORCE)
set(CPU_SNAP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CPU_SNAP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CPU_SNAP_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(CPU_SNAP_INSTALL OFF CACHE BOOL "" FORCE)

# SYSTEM: consumers see cpu-snap's include directories as system includes.
add_subdirectory(${PROJECT_SOURCE_DIR}/external/cpu-snap SYSTEM)

target_link_libraries(rocprofiler-systems-cpu-snap INTERFACE cpu_snap::sampling)
