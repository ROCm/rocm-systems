# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(gtest)

# ============================================================================ #
# Local schema headers for the unit tests
#
# The backend impl (sqlite_backend_impl.hpp) needs the rocpd schema either from
# the rocprofiler-sdk-rocpd library (which drags in real sqlite3) or from
# locally generated headers. The mock-based unit target must stay free of the
# real sqlite3 library, so it always uses the locally generated schema headers
# (USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=0), independent of the global setting.
# ============================================================================ #

set(UNIT_TEST_SCHEMA_DIR ${CMAKE_SOURCE_DIR}/source/data_storage/schema)
set(UNIT_TEST_SCHEMA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/schema_gen)
rocpd_configure_rocpd_schema_files(
    ${UNIT_TEST_SCHEMA_DIR}
    ${UNIT_TEST_SCHEMA_BINARY_DIR}
)

# ============================================================================ #
# Mock-based unit tests (NO real sqlite3, NO rocpd.db, NO profiler-hub library)
#
# Every test in this target runs against the gmock-backed mock_sqlite3 policy or
# exercises pure query/serialization logic. The target deliberately does not
# link profiler-hub or profiler-hub-sqlite3.
# ============================================================================ #

set(UNIT_TEST_SOURCES
    database_test.cpp
    mock_sqlite3_test.cpp
    insert_query_builders_test.cpp
    table_insert_query_test.cpp
    table_select_query_test.cpp
    entity_utility_test.cpp
    json_serializer_test.cpp
    # Compiled directly so the unit target needs no production library (and thus
    # no transitive sqlite3); json serialization has no sqlite dependency.
    ${CMAKE_SOURCE_DIR}/source/json_serializers.cpp
)

add_executable(profiler-hub_unit_tests ${UNIT_TEST_SOURCES})

target_link_libraries(
    profiler-hub_unit_tests
    PRIVATE
        profiler-hub_queries
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

target_include_directories(
    profiler-hub_unit_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/source
        ${CMAKE_SOURCE_DIR}/source/common
        ${CMAKE_SOURCE_DIR}/source/data_storage
        ${CMAKE_SOURCE_DIR}/source/queries
        ${CMAKE_BINARY_DIR}/include
        ${UNIT_TEST_SCHEMA_BINARY_DIR}
)

# Force the local (sqlite3-free) schema path inside sqlite_backend_impl.hpp.
target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=0
)

target_compile_features(profiler-hub_unit_tests PRIVATE cxx_std_17)

if(PROFILER_HUB_ENABLE_COVERAGE)
    profiler_hub_add_coverage_flags(profiler-hub_unit_tests)
endif()

gtest_discover_tests(
    profiler-hub_unit_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES LABELS "unit"
)
