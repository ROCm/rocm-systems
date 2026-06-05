# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ============================================================================ #
# Integration tests (real sqlite3 + rocpd.db fixture + profiler-hub library)
#
# Exercise the public reader/writer/backend API end-to-end against a real SQLite
# engine and a populated rocpd.db fixture, so they cannot use the mock seam.
# ============================================================================ #

include(gtest)

set(INTEGRATION_TEST_SOURCES writer_test.cpp reader_test.cpp backend_test.cpp)

add_executable(profiler-hub_integration_tests ${INTEGRATION_TEST_SOURCES})

# Link the STATIC profiler-hub so the backend's explicitly-instantiated members
# and the test's header-inline template instantiations share a SINGLE bundled
# SQLite copy inside one binary.
target_link_libraries(
    profiler-hub_integration_tests
    PRIVATE
        profiler-hub-static
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
    profiler-hub_integration_tests
    PUBLIC ROCPD_DB_PATH="${ROCPD_DB_PATH}"
)

target_include_directories(
    profiler-hub_integration_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/source
        ${CMAKE_SOURCE_DIR}/source/common
        ${CMAKE_SOURCE_DIR}/source/data_storage
        ${CMAKE_SOURCE_DIR}/source/queries
        ${CMAKE_BINARY_DIR}/include
)

target_compile_features(profiler-hub_integration_tests PRIVATE cxx_std_17)

if(PROFILER_HUB_ENABLE_COVERAGE)
    profiler_hub_add_coverage_flags(profiler-hub_integration_tests)
endif()

gtest_discover_tests(
    profiler-hub_integration_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES LABELS "integration"
)
