# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(trace_cache_tests_sources
    test_agent_info_json.cpp
    test_archive.cpp
    test_args_serialization.cpp
    test_buffer_storage.cpp
    test_cache_integration.cpp
    test_cacheable.cpp
    test_collection_sink.cpp
    test_counter_records.cpp
    test_counter_schema_table.cpp
    test_discovery.cpp
    test_enabled_formats.cpp
    test_flush_worker.cpp
    test_kfd_args.cpp
    test_metadata_registry_json.cpp
    test_metadata_registry.cpp
    test_record_format.cpp
    test_rocpd_counter_batch.cpp
    test_rocpd_processor.cpp
    test_rocpd_xcp_output.cpp
    test_sample_type.cpp
    test_storage_parser.cpp
    test_type_registry.cpp
    test_unified_memory_processor.cpp
)

add_library(trace-cache-tests OBJECT ${trace_cache_tests_sources})

# The rocpd counter tests read pmc_event rows with SQLite directly: profiler-hub's
# reader cannot return them yet. rocprof-sys-unit-tests links it too.
add_library(trace-cache-tests-sqlite INTERFACE)
if(TARGET profiler-hub-sqlite3)
    target_link_libraries(trace-cache-tests-sqlite INTERFACE profiler-hub-sqlite3)
else()
    find_package(SQLite3 REQUIRED)
    target_link_libraries(trace-cache-tests-sqlite INTERFACE SQLite::SQLite3)
endif()

target_link_libraries(
    trace-cache-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-profiler-hub
        rocprofiler-systems-logger
        rocprofiler-systems-headers
        rocprofiler-systems-rocm
        rocprofiler-systems-json
        rocprofiler-systems-timemory
        trace-cache-tests-sqlite
)

target_include_directories(
    trace-cache-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/core
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${CMAKE_BINARY_DIR}/source/lib/
)
