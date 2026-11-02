# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(instrument-tests OBJECT test_dynamic_dependency_listing.cpp)

target_include_directories(instrument-tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../..)

target_link_libraries(
    instrument-tests
    PUBLIC rocprofiler-systems-common-library rocprofiler-systems-googletest-library
)
