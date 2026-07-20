# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(process-tree-tests OBJECT process_tree_tests.cpp)

target_link_libraries(
    process-tree-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(process-tree-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(output-registry-tests OBJECT registry_tests.cpp)

target_link_libraries(
    output-registry-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(output-registry-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(summary-writer-tests OBJECT summary_writer_tests.cpp)

target_link_libraries(
    summary-writer-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
)

target_include_directories(summary-writer-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)
