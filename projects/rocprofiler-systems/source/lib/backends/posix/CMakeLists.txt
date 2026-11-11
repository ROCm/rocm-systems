# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocprofiler-systems-posix-backend INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-posix-backend
    ALIAS rocprofiler-systems-posix-backend
)

target_link_libraries(
    rocprofiler-systems-posix-backend
    INTERFACE rocprofiler-systems::rocprofiler-systems-headers
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
