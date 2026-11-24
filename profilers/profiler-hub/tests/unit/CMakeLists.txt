# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(gtest)

set(UNIT_TEST_SOURCES
    insert_query_builders_test.cpp
    table_insert_query_test.cpp
    table_select_query_test.cpp
    entity_utility_test.cpp
    json_serializer_test.cpp
    writer_test.cpp
    reader_test.cpp
    rocpd_v3_table_events_test.cpp
    rocpd_v3_correlated_events_test.cpp
    reader_id_range_test.cpp
    sqlite_backend_impl_test.cpp
    ph_ctx_test.cpp
    catalog_aggregate_store_test.cpp
    populate_reader_catalog_test.cpp
    c_api_test.cpp
    c_api_track_test.cpp
    c_api_future_test.cpp
    cached_track_reader_test.cpp
    connection_pool_test.cpp
    connection_source_test.cpp
    depth_tracker_test.cpp
    event_partitions_test.cpp
    natural_merge_sort_test.cpp
    thread_pool_test.cpp
    track_window_test.cpp
    track_file_store_test.cpp
)

# These tests read schema 2 traces, so they need the schema 2 reader compiled in.
if(PROFILER_HUB_ENABLE_ROCPD_V2)
    list(APPEND UNIT_TEST_SOURCES rocpd_v2_reader_test.cpp)
endif()

add_executable(profiler-hub_unit_tests ${UNIT_TEST_SOURCES})

# gtest's TLS symbols need explicit pthread linkage under amd-llvm (clang/lld), which
# gcc provided implicitly. Threads::Threads adapts: no flag where libc supplies pthread
# (glibc >= 2.34), -pthread on older sysroots.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(
    profiler-hub_unit_tests
    PRIVATE
        profiler-hub
        profiler-hub_queries
        profiler-hub-sqlite3
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        Threads::Threads
)

set(ROCPD_DB_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd.db)

target_compile_definitions(
    profiler-hub_unit_tests
    PUBLIC ROCPD_DB_PATH="${ROCPD_DB_PATH}"
)

# The library defines this privately. Tests that write schema 2 traces use their own
# copy to compile out when the schema 2 reader is off.
target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE PROFILER_HUB_ENABLE_ROCPD_V2=$<BOOL:${PROFILER_HUB_ENABLE_ROCPD_V2}>
)

if(PH_USE_NOMUTEX_OPTIMIZATION)
    target_compile_definitions(
        profiler-hub_unit_tests
        PRIVATE PH_USE_NOMUTEX_OPTIMIZATION
    )
endif()

# Same values storage_impl.cpp reports from get_storage_version(). Both come from
# the VERSION file via project(), so a version bump does not require a test edit.
target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:PROFILER_HUB_VERSION_MAJOR=${profiler-hub_VERSION_MAJOR}>
        $<$<COMPILE_LANGUAGE:CXX>:PROFILER_HUB_VERSION_MINOR=${profiler-hub_VERSION_MINOR}>
        $<$<COMPILE_LANGUAGE:CXX>:PROFILER_HUB_VERSION_PATCH=${profiler-hub_VERSION_PATCH}>
)

add_dependencies(profiler-hub_unit_tests profiler-hub)

target_include_directories(
    profiler-hub_unit_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/source
        ${CMAKE_SOURCE_DIR}/source/common
        ${CMAKE_SOURCE_DIR}/source/data_storage
        ${CMAKE_SOURCE_DIR}/source/queries
        ${SQL_SCHEMA_BINARY_DIR}
)

target_compile_features(profiler-hub_unit_tests PRIVATE cxx_std_20)

# Several tests (e.g. sqlite_backend_impl_test.cpp) #include headers with LOG_*
# calls directly, compiling them fresh into this TU rather than reusing
# profiler-hub-objects' copy, so this target needs its own copy of the define.
if(PROFILER_HUB_ENABLE_LOGGING)
    target_compile_definitions(
        profiler-hub_unit_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:PROFILER_HUB_ENABLE_LOGGING=1>
    )
else()
    target_compile_definitions(
        profiler-hub_unit_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:PROFILER_HUB_ENABLE_LOGGING=0>
    )
endif()

if(PROFILER_HUB_ENABLE_COVERAGE)
    profiler_hub_add_coverage_flags(profiler-hub_unit_tests)
endif()

gtest_discover_tests(
    profiler-hub_unit_tests
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES LABELS "unit"
)
