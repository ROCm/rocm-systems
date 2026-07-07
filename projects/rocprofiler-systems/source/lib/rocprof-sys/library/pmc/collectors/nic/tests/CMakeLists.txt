# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(pmc-nic-collector-tests OBJECT test_nic_device.cpp)

target_link_libraries(
    pmc-nic-collector-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-logger
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-rocm
        rocprofiler-systems::rocprofiler-systems-compile-definitions
)

if(ROCPROFSYS_BUILD_AINIC)
    target_compile_definitions(pmc-nic-collector-tests PUBLIC ROCPROFSYS_BUILD_AINIC=1)
endif()

target_include_directories(
    pmc-nic-collector-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${PROJECT_BINARY_DIR}/source/lib
)
