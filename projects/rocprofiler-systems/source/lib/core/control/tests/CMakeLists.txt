# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(control_tests_sources test_session.cpp)

add_library(control-tests OBJECT ${control_tests_sources})

target_link_libraries(
    control-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
)

target_include_directories(
    control-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib/
)
