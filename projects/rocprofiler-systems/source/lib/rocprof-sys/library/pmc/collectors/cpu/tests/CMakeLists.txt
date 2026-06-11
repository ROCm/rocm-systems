# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

add_library(
    pmc-cpu-collector-tests
    OBJECT
    test_device.cpp
    test_sample.cpp
    test_rocpd_round_trip.cpp
)

target_link_libraries(
    pmc-cpu-collector-tests
    PRIVATE
        rocprofiler-systems-pmc-library
        rocprofiler-systems-googletest-library
        rocprofiler-systems-logger
        rocprofiler-systems-common-library
        rocprofiler-systems-user-library
        rocprofiler-systems::rocprofiler-systems-headers
)

target_include_directories(
    pmc-cpu-collector-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
)
