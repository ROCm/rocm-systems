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
)

if(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
    target_link_libraries(
        profiler-hub_unit_tests
        PRIVATE rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd
    )

    target_compile_definitions(
        profiler-hub_unit_tests
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=1>
    )
else()
    target_include_directories(
        profiler-hub_unit_tests
        PRIVATE ${SQL_SCHEMA_BINARY_DIR}
    )

    target_compile_definitions(
        profiler-hub_unit_tests
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=0>
    )
endif()

target_compile_features(profiler-hub_unit_tests PRIVATE cxx_std_17)

if(PROFILER_HUB_ENABLE_COVERAGE)
    profiler_hub_add_coverage_flags(profiler-hub_unit_tests)
endif()

# ASAN: TheRock forces -shared-libsan fleet-wide but amd-llvm ships only the static asan
# runtime, so link it statically here.
if(
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    AND CMAKE_CXX_FLAGS MATCHES "-fsanitize=address"
)
    target_link_options(profiler-hub_unit_tests PRIVATE -static-libsan)
endif()

# Synthetic v3 edge-matrix fixture, generated at build time from committed SQL.
# The v3 schema is canonical at source/data_storage/schema/rocpd_tables.sql, so we
# reference it directly to avoid a drifting duplicate. The {{uuid}}/{{guid}}
# placeholders are substituted exactly as get_schema_query() does, then fed
# through the sqlite3 CLI. The db has no rocpd_timestamp table, so the reader
# selects the v3 backend. It seeds the two degenerate (bare non-pmc sample track)
# and the six writer-id-blocked (rocpd_arg on kd/mc/ma; counter/track identity)
# edge tests that cannot be expressed through the current writer API. Regenerate:
#   uuid=00000000000000000000000000e30003
#   sed -e "s/{{uuid}}/_$uuid/g" -e "s/{{guid}}/$uuid/g" \
#       ../../source/data_storage/schema/rocpd_tables.sql \
#       fixtures/rocpd_v3_edge_data.sql | sqlite3 rocpd_v3_edge.db
find_program(SQLITE3_EXECUTABLE sqlite3)
if(NOT SQLITE3_EXECUTABLE)
    message(
        FATAL_ERROR
        "sqlite3 CLI not found; it is required to build the synthetic v3 edge "
        "test fixture (tests/unit/fixtures/rocpd_v3_edge_data.sql)."
    )
endif()

set(_v3e_uuid "00000000000000000000000000e30003")
set(_v3e_schema_src
    ${CMAKE_SOURCE_DIR}/source/data_storage/schema/rocpd_tables.sql
)
set(_v3e_data_src ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v3_edge_data.sql)
set(_v3e_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v3_edge_gen.sql)
file(READ ${_v3e_schema_src} _v3e_schema)
file(READ ${_v3e_data_src} _v3e_data)
# The canonical schema ends with an INSERT INTO rocpd_metadata that uses
# double-quoted string literals. The modern sqlite3 CLI rejects double-quoted
# strings in DML (only DDL keeps the legacy behavior), and the reader never reads
# metadata anyway (v3 is detected by the ABSENCE of rocpd_timestamp). Strip that
# single trailing DML statement so the CLI is fed pure DDL. [^;] spans newlines.
string(
    REGEX REPLACE
    "INSERT INTO[^;]*rocpd_metadata[^;]*;"
    ""
    _v3e_schema
    "${_v3e_schema}"
)
string(REPLACE "{{uuid}}" "_${_v3e_uuid}" _v3e_schema "${_v3e_schema}")
string(REPLACE "{{guid}}" "${_v3e_uuid}" _v3e_schema "${_v3e_schema}")
string(REPLACE "{{uuid}}" "_${_v3e_uuid}" _v3e_data "${_v3e_data}")
string(REPLACE "{{guid}}" "${_v3e_uuid}" _v3e_data "${_v3e_data}")
file(WRITE ${_v3e_gen_sql} "${_v3e_schema}\n${_v3e_data}\n")

set(ROCPD_DB_V3_EDGE_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v3_edge.db)
add_custom_command(
    OUTPUT ${ROCPD_DB_V3_EDGE_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V3_EDGE_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V3_EDGE_PATH} ".read ${_v3e_gen_sql}"
    DEPENDS ${_v3e_gen_sql} ${_v3e_schema_src} ${_v3e_data_src}
    COMMENT "Generating synthetic v3 edge-matrix test fixture"
    VERBATIM
)
add_custom_target(profiler-hub_v3_edge_fixture DEPENDS ${ROCPD_DB_V3_EDGE_PATH})
add_dependencies(profiler-hub_unit_tests profiler-hub_v3_edge_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V3_EDGE_PATH="${ROCPD_DB_V3_EDGE_PATH}"
)

gtest_discover_tests(
    profiler-hub_unit_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES LABELS "unit"
)
