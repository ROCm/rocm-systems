# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(units_tests_sources test_units_typed.cpp)

add_library(lib-common-units-tests OBJECT ${units_tests_sources})

target_link_libraries(
    lib-common-units-tests
    PRIVATE rocprofiler-systems-common-library rocprofiler-systems-googletest-library
)

target_include_directories(
    lib-common-units-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib
)
