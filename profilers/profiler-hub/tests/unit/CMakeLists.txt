# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(gtest)

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/rocpd.db
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd.db
    COPYONLY
)

set(ROCPD_DB_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd.db)

function(profiler_hub_configure_unit_tests target)
    target_compile_definitions(
        ${target}
        PUBLIC ROCPD_DB_PATH="${ROCPD_DB_PATH}"
    )

    target_include_directories(
        ${target}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/source
            ${CMAKE_SOURCE_DIR}/source/common
            ${CMAKE_SOURCE_DIR}/source/data_storage
            ${CMAKE_SOURCE_DIR}/source/queries
    )

    # backend_over_mock_test.cpp includes sqlite_backend_impl.hpp directly to
    # instantiate database_backend over the mock policy, so unit-test targets
    # need the same schema configuration as profiler-hub-objects.
    if(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
        target_link_libraries(
            ${target}
            PRIVATE rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd
        )
        target_compile_definitions(
            ${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=1>
        )
    else()
        target_include_directories(${target} PRIVATE ${SQL_SCHEMA_BINARY_DIR})
        target_compile_definitions(
            ${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=0>
        )
    endif()

    target_compile_features(${target} PRIVATE cxx_std_17)

    if(PROFILER_HUB_ENABLE_COVERAGE)
        profiler_hub_add_coverage_flags(${target})
    endif()

    gtest_discover_tests(
        ${target}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        PROPERTIES LABELS "unit"
    )
endfunction()

set(UNIT_TEST_SOURCES
    writer_test.cpp
    reader_test.cpp
    insert_query_builders_test.cpp
    table_insert_query_test.cpp
    table_select_query_test.cpp
    entity_utility_test.cpp
    json_serializer_test.cpp
    mock_sqlite3_test.cpp
    backend_over_mock_test.cpp
)

# Build-artifact regression guard: verify libprofiler-hub.so does not export the
# bundled sqlite3_* symbols
if(NOT PROFILER_HUB_SQLITE3_USE_SYSTEM)
    list(APPEND UNIT_TEST_SOURCES symbol_seal_test.cpp)
endif()

add_executable(profiler-hub_unit_tests ${UNIT_TEST_SOURCES})

# writer_test.cpp verifies DB contents via direct SQLite queries on disk files.
# Use the system library so the test binary does not embed a second bundled copy
# alongside libprofiler-hub.so.
find_package(SQLite3 REQUIRED)

target_link_libraries(
    profiler-hub_unit_tests
    PRIVATE
        profiler-hub
        profiler-hub_queries
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        SQLite::SQLite3
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

if(NOT PROFILER_HUB_SQLITE3_USE_SYSTEM)
    target_compile_definitions(
        profiler-hub_unit_tests
        PRIVATE PROFILER_HUB_SHARED_LIB="$<TARGET_FILE:profiler-hub>"
    )
    add_dependencies(profiler-hub_unit_tests profiler-hub)
endif()

profiler_hub_configure_unit_tests(profiler-hub_unit_tests)

# database_test.cpp includes database_backend templates into the test binary.
# Keep it on a static link so SQLite handles are not split across libprofiler-hub.so
# and the executable. TODO: consolidate into profiler-hub_unit_tests once backend
# tests move to the public API (storage/writer/reader) or mock-only coverage.
add_executable(profiler-hub_sqlite_backend_tests database_test.cpp)

target_link_libraries(
    profiler-hub_sqlite_backend_tests
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

profiler_hub_configure_unit_tests(profiler-hub_sqlite_backend_tests)
