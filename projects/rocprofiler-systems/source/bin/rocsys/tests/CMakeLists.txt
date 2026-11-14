# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocsys-tests OBJECT test_cli_dispatcher.cpp)

target_link_libraries(
    rocsys-tests
    PUBLIC
        rocprofiler-systems::rocprofiler-systems-rocsys-core
        rocprofiler-systems-googletest-library
)
