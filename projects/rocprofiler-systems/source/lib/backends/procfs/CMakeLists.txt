# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocprofiler-systems-procfs-backend INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-procfs-backend
    ALIAS rocprofiler-systems-procfs-backend
)

target_link_libraries(
    rocprofiler-systems-procfs-backend
    INTERFACE rocprofiler-systems::rocprofiler-systems-headers rocprofiler-systems-logger
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
