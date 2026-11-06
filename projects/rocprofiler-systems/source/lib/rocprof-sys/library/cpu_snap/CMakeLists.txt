# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(cpu_snap_sources ${CMAKE_CURRENT_LIST_DIR}/sampling.cpp)

set(cpu_snap_headers
    ${CMAKE_CURRENT_LIST_DIR}/config.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sampling.hpp
)

target_sources(
    rocprofiler-systems-object-library
    PRIVATE ${cpu_snap_sources} ${cpu_snap_headers}
)

# Object files carry no link dependencies: the libraries built from them link
# rocprofiler-systems-cpu-snap too (source/lib/rocprof-sys/CMakeLists.txt).
target_link_libraries(
    rocprofiler-systems-object-library
    PRIVATE rocprofiler-systems::rocprofiler-systems-cpu-snap
)
