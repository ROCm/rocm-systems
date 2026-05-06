# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(trace_cache_tests_sources
    test_type_registry.cpp
    test_cacheable.cpp
    test_buffer_storage.cpp
    test_storage_parser.cpp
    test_flush_worker.cpp
    test_cache_integration.cpp
    test_sample_type.cpp
    test_kfd_args.cpp
    test_rocpd_xcp_output.cpp
    test_xgmi_per_link_round_trip.cpp
)

add_library(trace-cache-tests OBJECT ${trace_cache_tests_sources})

target_link_libraries(
    trace-cache-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
)

target_include_directories(
    trace-cache-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${CMAKE_BINARY_DIR}/source/lib/
)

# test_xgmi_per_link_round_trip.cpp inspects the production source files to
# guard against name-format drift; pin the project source root so the test
# locates them deterministically.
set_source_files_properties(
    test_xgmi_per_link_round_trip.cpp
    PROPERTIES COMPILE_DEFINITIONS "TEST_SOURCE_ROOT=\"${PROJECT_SOURCE_DIR}\""
)
