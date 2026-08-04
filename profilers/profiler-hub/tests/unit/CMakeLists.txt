# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(gtest)

set(UNIT_TEST_SOURCES
    insert_query_builders_test.cpp
    table_insert_query_test.cpp
    table_select_query_test.cpp
    entity_utility_test.cpp
    json_serializer_test.cpp
    symbol_seal_test.cpp
    writer_test.cpp
    reader_test.cpp
    sqlite_backend_impl_test.cpp
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
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

set(ROCPD_DB_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd.db)

target_compile_definitions(
    profiler-hub_unit_tests
    PUBLIC ROCPD_DB_PATH="${ROCPD_DB_PATH}"
)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE PROFILER_HUB_SHARED_LIB="$<TARGET_FILE:profiler-hub>"
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

target_compile_features(profiler-hub_unit_tests PRIVATE cxx_std_17)

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
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES LABELS "unit"
)
