# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(demangler-tests OBJECT demangler_test.cpp)

target_link_libraries(
    demangler-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(demangler-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(output-summary-tests OBJECT output_summary_tests.cpp)

target_link_libraries(
    output-summary-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems::rocprofiler-systems-perfetto
)

target_include_directories(output-summary-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)
