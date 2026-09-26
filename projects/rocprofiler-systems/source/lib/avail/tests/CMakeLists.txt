# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(avail-catalog-tests OBJECT test_catalog.cpp)

target_link_libraries(
    avail-catalog-tests
    PUBLIC
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-avail-library
)
