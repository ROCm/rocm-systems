# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(demangler-tests OBJECT demangler_test.cpp)

target_link_libraries(
    demangler-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(demangler-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(track-registry-tests OBJECT test_track_registry.cpp)

target_link_libraries(
    track-registry-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(track-registry-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(output-format-tests OBJECT test_output_format.cpp)

target_link_libraries(
    output-format-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
        rocprofiler-systems-headers
        rocprofiler-systems-timemory
)

target_include_directories(
    output-format-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib/
)

add_library(parse-numeric-range-tests OBJECT test_parse_numeric_range.cpp)

target_link_libraries(
    parse-numeric-range-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
        rocprofiler-systems-headers
        rocprofiler-systems-timemory
)

target_include_directories(
    parse-numeric-range-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib/
)

add_library(rank-passes-filter-tests OBJECT test_rank_passes_filter.cpp)

target_link_libraries(
    rank-passes-filter-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
        rocprofiler-systems-headers
        rocprofiler-systems-timemory
)

target_include_directories(
    rank-passes-filter-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib/
)
