# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    argparse-tests
    OBJECT
    test_core_flags_table.cpp
    test_flag_descriptor.cpp
    test_parsed_values.cpp
    test_register_group.cpp
)

target_link_libraries(
    argparse-tests
    PUBLIC
        rocprofiler-systems-headers
        rocprofiler-systems-googletest-library
        rocprofiler-systems-timemory
        rocprofiler-systems-common-library
        rocprofiler-systems-core-library
)

target_include_directories(
    argparse-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/core
)
