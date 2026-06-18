# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(gtest)

set(UNIT_TEST_SOURCES
    writer_test.cpp
    reader_test.cpp
    insert_query_builders_test.cpp
    table_insert_query_test.cpp
    table_select_query_test.cpp
    database_test.cpp
    entity_utility_test.cpp
    json_serializer_test.cpp
)

add_executable(profiler-hub_unit_tests ${UNIT_TEST_SOURCES})

target_link_libraries(
    profiler-hub_unit_tests
    PRIVATE
        profiler-hub
        profiler-hub_queries
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        profiler-hub-sqlite3
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/rocpd.db
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd.db
    COPYONLY
)

set(ROCPD_DB_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd.db)

target_compile_definitions(
    profiler-hub_unit_tests
    PUBLIC ROCPD_DB_PATH="${ROCPD_DB_PATH}"
)

target_include_directories(
    profiler-hub_unit_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/source
        ${CMAKE_SOURCE_DIR}/source/common
        ${CMAKE_SOURCE_DIR}/source/data_storage
        ${CMAKE_SOURCE_DIR}/source/queries
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
