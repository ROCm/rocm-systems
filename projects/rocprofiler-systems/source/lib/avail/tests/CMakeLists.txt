# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(avail-tests OBJECT test_devices.cpp test_gpu_counters.cpp test_traces.cpp)

target_link_libraries(
    avail-tests
    PUBLIC
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-avail
)
