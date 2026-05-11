# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    diagnostic-tests
    OBJECT
    test_color.cpp
    test_stacktrace.cpp
    test_format_exception.cpp
)

target_link_libraries(
    diagnostic-tests
    PUBLIC
        rocprofiler-systems::rocprofiler-systems-diagnostic
        rocprofiler-systems-common-library
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-interface-library
)
