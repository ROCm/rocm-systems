# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(trace_cache_tests_sources
    test_buffer_storage.cpp
    test_cache_integration.cpp
    test_cacheable.cpp
    test_discovery.cpp
    test_enabled_formats.cpp
    test_flush_worker.cpp
    test_kfd_args.cpp
    test_rocpd_xcp_output.cpp
    test_sample_type.cpp
    test_storage_parser.cpp
    test_type_registry.cpp
    test_unified_memory_processor.cpp
)

add_library(trace-cache-tests OBJECT ${trace_cache_tests_sources})

target_link_libraries(
    trace-cache-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
        rocprofiler-systems-ghc-filesystem
        rocprofiler-systems-headers
        rocprofiler-systems-rocm
        rocprofiler-systems-json
        rocprofiler-systems-timemory
)

target_include_directories(
    trace-cache-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${CMAKE_BINARY_DIR}/source/lib/
)
