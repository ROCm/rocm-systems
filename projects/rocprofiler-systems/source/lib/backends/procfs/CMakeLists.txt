# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocprofiler-systems-procfs-backend STATIC backend.cpp backend.hpp)
add_library(
    rocprofiler-systems::rocprofiler-systems-procfs-backend
    ALIAS rocprofiler-systems-procfs-backend
)

set_target_properties(rocprofiler-systems-procfs-backend PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
    rocprofiler-systems-procfs-backend
    PUBLIC rocprofiler-systems::rocprofiler-systems-headers rocprofiler-systems-logger
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
