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
    reader_core_test.cpp
    reader_flows_test.cpp
    reader_pmc_test.cpp
    reader_memory_test.cpp
    reader_summary_test.cpp
    reader_track_metadata_test.cpp
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

# Synthetic v3 edge-matrix fixture: seeds the two degenerate (bare non-pmc sample
# track) and six writer-id-blocked (rocpd_arg on kd/mc/ma; counter/track identity)
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

# Synthetic v3 kernel_dispatch_pmc fixture: encodes two PMC types across three
# kernel_dispatch rows so both track discovery and get_interval_track() can be
# asserted on real values.
set(_v3p_uuid "0000000000000000000000000b030003")
set(_v3p_schema_src
    ${CMAKE_SOURCE_DIR}/source/data_storage/schema/rocpd_tables.sql
)
set(_v3p_data_src ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v3_kd_pmc_data.sql)
set(_v3p_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v3_kd_pmc_gen.sql)
file(READ ${_v3p_schema_src} _v3p_schema)
file(READ ${_v3p_data_src} _v3p_data)
string(
    REGEX REPLACE
    "INSERT INTO[^;]*rocpd_metadata[^;]*;"
    ""
    _v3p_schema
    "${_v3p_schema}"
)
string(REPLACE "{{uuid}}" "_${_v3p_uuid}" _v3p_schema "${_v3p_schema}")
string(REPLACE "{{guid}}" "${_v3p_uuid}" _v3p_schema "${_v3p_schema}")
string(REPLACE "{{uuid}}" "_${_v3p_uuid}" _v3p_data "${_v3p_data}")
string(REPLACE "{{guid}}" "${_v3p_uuid}" _v3p_data "${_v3p_data}")
file(WRITE ${_v3p_gen_sql} "${_v3p_schema}\n${_v3p_data}\n")

set(ROCPD_DB_V3_KD_PMC_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v3_kd_pmc.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V3_KD_PMC_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V3_KD_PMC_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V3_KD_PMC_PATH} ".read ${_v3p_gen_sql}"
    DEPENDS ${_v3p_gen_sql} ${_v3p_schema_src} ${_v3p_data_src}
    COMMENT "Generating synthetic v3 kernel_dispatch_pmc test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v3_kd_pmc_fixture
    DEPENDS ${ROCPD_DB_V3_KD_PMC_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v3_kd_pmc_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V3_KD_PMC_PATH="${ROCPD_DB_V3_KD_PMC_PATH}"
)

# Synthetic v3 track-shapes fixture: carries one track of each v3 dma / memory /
# cpu_thread shape the other v3 fixtures leave dark in get_track_stats /
# get_interval_track.
set(_v3s_uuid "00000000000000000000000000730003")
set(_v3s_schema_src
    ${CMAKE_SOURCE_DIR}/source/data_storage/schema/rocpd_tables.sql
)
set(_v3s_data_src
    ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v3_track_shapes_data.sql
)
set(_v3s_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v3_track_shapes_gen.sql)
file(READ ${_v3s_schema_src} _v3s_schema)
file(READ ${_v3s_data_src} _v3s_data)
string(
    REGEX REPLACE
    "INSERT INTO[^;]*rocpd_metadata[^;]*;"
    ""
    _v3s_schema
    "${_v3s_schema}"
)
string(REPLACE "{{uuid}}" "_${_v3s_uuid}" _v3s_schema "${_v3s_schema}")
string(REPLACE "{{guid}}" "${_v3s_uuid}" _v3s_schema "${_v3s_schema}")
string(REPLACE "{{uuid}}" "_${_v3s_uuid}" _v3s_data "${_v3s_data}")
string(REPLACE "{{guid}}" "${_v3s_uuid}" _v3s_data "${_v3s_data}")
file(WRITE ${_v3s_gen_sql} "${_v3s_schema}\n${_v3s_data}\n")

set(ROCPD_DB_V3_TRACK_SHAPES_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v3_track_shapes.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V3_TRACK_SHAPES_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V3_TRACK_SHAPES_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V3_TRACK_SHAPES_PATH}
        ".read ${_v3s_gen_sql}"
    DEPENDS ${_v3s_gen_sql} ${_v3s_schema_src} ${_v3s_data_src}
    COMMENT "Generating synthetic v3 track-shapes test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v3_track_shapes_fixture
    DEPENDS ${ROCPD_DB_V3_TRACK_SHAPES_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v3_track_shapes_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V3_TRACK_SHAPES_PATH="${ROCPD_DB_V3_TRACK_SHAPES_PATH}"
)

# Synthetic v4.0 kernel_dispatch_pmc fixture. No canonical v4.0 DDL lives
# in-tree; the schema copy ships in fixtures/. Mirrors the v3 fixture's data
# shape.
set(_v4p_uuid "0000000000000000000000000b040004")
set(_v4p_data_src ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4_kd_pmc_data.sql)
set(_v4p_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v4_kd_pmc_gen.sql)
file(READ ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql _v4p_schema)
file(READ ${_v4p_data_src} _v4p_data)
string(REPLACE "{{uuid}}" "_${_v4p_uuid}" _v4p_schema "${_v4p_schema}")
string(REPLACE "{{guid}}" "${_v4p_uuid}" _v4p_schema "${_v4p_schema}")
string(REPLACE "{{uuid}}" "_${_v4p_uuid}" _v4p_data "${_v4p_data}")
string(REPLACE "{{guid}}" "${_v4p_uuid}" _v4p_data "${_v4p_data}")
file(WRITE ${_v4p_gen_sql} "${_v4p_schema}\n${_v4p_data}\n")

set(ROCPD_DB_V4_KD_PMC_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v4_kd_pmc.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V4_KD_PMC_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V4_KD_PMC_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V4_KD_PMC_PATH} ".read ${_v4p_gen_sql}"
    DEPENDS
        ${_v4p_gen_sql}
        ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql
        ${_v4p_data_src}
    COMMENT "Generating synthetic v4.0 kernel_dispatch_pmc test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v4_kd_pmc_fixture
    DEPENDS ${ROCPD_DB_V4_KD_PMC_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v4_kd_pmc_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V4_KD_PMC_PATH="${ROCPD_DB_V4_KD_PMC_PATH}"
)

# Synthetic v4.0 counter/scalar fixture. No real v4.0 capture available to the
# project contains rocpd_sample rows, so this is the only way to positively
# exercise the v4.0 scalar/counter path.
set(_v4c_uuid "00000000000000000000000000c00001")
set(_v4c_data_src ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4_counter_data.sql)
set(_v4c_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v4_counter_gen.sql)
file(READ ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql _v4c_schema)
file(READ ${_v4c_data_src} _v4c_data)
string(REPLACE "{{uuid}}" "_${_v4c_uuid}" _v4c_schema "${_v4c_schema}")
string(REPLACE "{{guid}}" "${_v4c_uuid}" _v4c_schema "${_v4c_schema}")
string(REPLACE "{{uuid}}" "_${_v4c_uuid}" _v4c_data "${_v4c_data}")
string(REPLACE "{{guid}}" "${_v4c_uuid}" _v4c_data "${_v4c_data}")
file(WRITE ${_v4c_gen_sql} "${_v4c_schema}\n${_v4c_data}\n")

set(ROCPD_DB_V4_COUNTER_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v4_counter.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V4_COUNTER_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V4_COUNTER_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V4_COUNTER_PATH}
        ".read ${_v4c_gen_sql}"
    DEPENDS
        ${_v4c_gen_sql}
        ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql
        ${_v4c_data_src}
    COMMENT "Generating synthetic v4.0 counter/scalar test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v4_counter_fixture
    DEPENDS ${ROCPD_DB_V4_COUNTER_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v4_counter_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V4_COUNTER_PATH="${ROCPD_DB_V4_COUNTER_PATH}"
)

# Synthetic v4.0 memory_activity fixture. No real v4.0 capture available to the
# project carries the memory_activity shape, so this is the only way to
# positively exercise the v4.0 memory-activity scalar/interval path.
set(_v4m_uuid "0000000000000000000000000c020002")
set(_v4m_data_src
    ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4_mem_activity_data.sql
)
set(_v4m_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v4_mem_activity_gen.sql)
file(READ ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql _v4m_schema)
file(READ ${_v4m_data_src} _v4m_data)
string(REPLACE "{{uuid}}" "_${_v4m_uuid}" _v4m_schema "${_v4m_schema}")
string(REPLACE "{{guid}}" "${_v4m_uuid}" _v4m_schema "${_v4m_schema}")
string(REPLACE "{{uuid}}" "_${_v4m_uuid}" _v4m_data "${_v4m_data}")
string(REPLACE "{{guid}}" "${_v4m_uuid}" _v4m_data "${_v4m_data}")
file(WRITE ${_v4m_gen_sql} "${_v4m_schema}\n${_v4m_data}\n")

set(ROCPD_DB_V4_MEM_ACTIVITY_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v4_mem_activity.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V4_MEM_ACTIVITY_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V4_MEM_ACTIVITY_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V4_MEM_ACTIVITY_PATH}
        ".read ${_v4m_gen_sql}"
    DEPENDS
        ${_v4m_gen_sql}
        ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql
        ${_v4m_data_src}
    COMMENT "Generating synthetic v4.0 memory_activity test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v4_mem_activity_fixture
    DEPENDS ${ROCPD_DB_V4_MEM_ACTIVITY_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v4_mem_activity_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V4_MEM_ACTIVITY_PATH="${ROCPD_DB_V4_MEM_ACTIVITY_PATH}"
)

# Synthetic v3 memory_activity fixture: exercises get_scalar_track(memory_activity)
# with a cumulative per-agent running sum and FREE agent_id recovery via address
# self-join.
set(_v3m_uuid "0000000000000000000000000c010001")
set(_v3m_schema_src
    ${CMAKE_SOURCE_DIR}/source/data_storage/schema/rocpd_tables.sql
)
set(_v3m_data_src
    ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v3_mem_activity_data.sql
)
set(_v3m_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v3_mem_activity_gen.sql)
file(READ ${_v3m_schema_src} _v3m_schema)
file(READ ${_v3m_data_src} _v3m_data)
string(
    REGEX REPLACE
    "INSERT INTO[^;]*rocpd_metadata[^;]*;"
    ""
    _v3m_schema
    "${_v3m_schema}"
)
string(REPLACE "{{uuid}}" "_${_v3m_uuid}" _v3m_schema "${_v3m_schema}")
string(REPLACE "{{guid}}" "${_v3m_uuid}" _v3m_schema "${_v3m_schema}")
string(REPLACE "{{uuid}}" "_${_v3m_uuid}" _v3m_data "${_v3m_data}")
string(REPLACE "{{guid}}" "${_v3m_uuid}" _v3m_data "${_v3m_data}")
file(WRITE ${_v3m_gen_sql} "${_v3m_schema}\n${_v3m_data}\n")

set(ROCPD_DB_V3_MEM_ACTIVITY_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v3_mem_activity.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V3_MEM_ACTIVITY_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V3_MEM_ACTIVITY_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V3_MEM_ACTIVITY_PATH}
        ".read ${_v3m_gen_sql}"
    DEPENDS ${_v3m_gen_sql} ${_v3m_schema_src} ${_v3m_data_src}
    COMMENT "Generating synthetic v3 memory_activity test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v3_mem_activity_fixture
    DEPENDS ${ROCPD_DB_V3_MEM_ACTIVITY_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v3_mem_activity_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V3_MEM_ACTIVITY_PATH="${ROCPD_DB_V3_MEM_ACTIVITY_PATH}"
)

# Synthetic v3 memory_activity time-window fixture: ALLOC/FREE rows straddle a
# chosen window so every time-window `continue` filter in
# get_scalar_track(memory_activity) fires.
set(_v3mw_uuid "00000000000000000000000000730045")
set(_v3mw_schema_src
    ${CMAKE_SOURCE_DIR}/source/data_storage/schema/rocpd_tables.sql
)
set(_v3mw_data_src
    ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v3_mem_activity_window_data.sql
)
set(_v3mw_gen_sql
    ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v3_mem_activity_window_gen.sql
)
file(READ ${_v3mw_schema_src} _v3mw_schema)
file(READ ${_v3mw_data_src} _v3mw_data)
string(
    REGEX REPLACE
    "INSERT INTO[^;]*rocpd_metadata[^;]*;"
    ""
    _v3mw_schema
    "${_v3mw_schema}"
)
string(REPLACE "{{uuid}}" "_${_v3mw_uuid}" _v3mw_schema "${_v3mw_schema}")
string(REPLACE "{{guid}}" "${_v3mw_uuid}" _v3mw_schema "${_v3mw_schema}")
string(REPLACE "{{uuid}}" "_${_v3mw_uuid}" _v3mw_data "${_v3mw_data}")
string(REPLACE "{{guid}}" "${_v3mw_uuid}" _v3mw_data "${_v3mw_data}")
file(WRITE ${_v3mw_gen_sql} "${_v3mw_schema}\n${_v3mw_data}\n")

set(ROCPD_DB_V3_MEM_ACTIVITY_WINDOW_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v3_mem_activity_window.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V3_MEM_ACTIVITY_WINDOW_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V3_MEM_ACTIVITY_WINDOW_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V3_MEM_ACTIVITY_WINDOW_PATH}
        ".read ${_v3mw_gen_sql}"
    DEPENDS ${_v3mw_gen_sql} ${_v3mw_schema_src} ${_v3mw_data_src}
    COMMENT "Generating synthetic v3 memory_activity time-window test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v3_mem_activity_window_fixture
    DEPENDS ${ROCPD_DB_V3_MEM_ACTIVITY_WINDOW_PATH}
)
add_dependencies(
    profiler-hub_unit_tests
    profiler-hub_v3_mem_activity_window_fixture
)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE
        ROCPD_DB_V3_MEM_ACTIVITY_WINDOW_PATH="${ROCPD_DB_V3_MEM_ACTIVITY_WINDOW_PATH}"
)

# Synthetic v4.0 summary/GROUP-BY-name fixture: shares its by-construction oracle
# with the v3 summary fixture so both reader backends assert identical aggregates.
set(_v4s_uuid "00000000000000000000000000740050")
set(_v4s_data_src ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4_summary_data.sql)
set(_v4s_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v4_summary_gen.sql)
file(READ ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql _v4s_schema)
file(READ ${_v4s_data_src} _v4s_data)
string(REPLACE "{{uuid}}" "_${_v4s_uuid}" _v4s_schema "${_v4s_schema}")
string(REPLACE "{{guid}}" "${_v4s_uuid}" _v4s_schema "${_v4s_schema}")
string(REPLACE "{{uuid}}" "_${_v4s_uuid}" _v4s_data "${_v4s_data}")
string(REPLACE "{{guid}}" "${_v4s_uuid}" _v4s_data "${_v4s_data}")
file(WRITE ${_v4s_gen_sql} "${_v4s_schema}\n${_v4s_data}\n")

set(ROCPD_DB_V4_SUMMARY_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v4_summary.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V4_SUMMARY_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V4_SUMMARY_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V4_SUMMARY_PATH}
        ".read ${_v4s_gen_sql}"
    DEPENDS
        ${_v4s_gen_sql}
        ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql
        ${_v4s_data_src}
    COMMENT "Generating synthetic v4.0 summary test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v4_summary_fixture
    DEPENDS ${ROCPD_DB_V4_SUMMARY_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v4_summary_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V4_SUMMARY_PATH="${ROCPD_DB_V4_SUMMARY_PATH}"
)

# Synthetic v3 summary/GROUP-BY-name fixture: shares its by-construction oracle
# with the v4 summary fixture so both reader backends assert identical aggregates.
set(_v3s_uuid "00000000000000000000000000730050")
set(_v3s_schema_src
    ${CMAKE_SOURCE_DIR}/source/data_storage/schema/rocpd_tables.sql
)
set(_v3s_data_src ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v3_summary_data.sql)
set(_v3s_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v3_summary_gen.sql)
file(READ ${_v3s_schema_src} _v3s_schema)
file(READ ${_v3s_data_src} _v3s_data)
string(
    REGEX REPLACE
    "INSERT INTO[^;]*rocpd_metadata[^;]*;"
    ""
    _v3s_schema
    "${_v3s_schema}"
)
string(REPLACE "{{uuid}}" "_${_v3s_uuid}" _v3s_schema "${_v3s_schema}")
string(REPLACE "{{guid}}" "${_v3s_uuid}" _v3s_schema "${_v3s_schema}")
string(REPLACE "{{uuid}}" "_${_v3s_uuid}" _v3s_data "${_v3s_data}")
string(REPLACE "{{guid}}" "${_v3s_uuid}" _v3s_data "${_v3s_data}")
file(WRITE ${_v3s_gen_sql} "${_v3s_schema}\n${_v3s_data}\n")

set(ROCPD_DB_V3_SUMMARY_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v3_summary.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V3_SUMMARY_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V3_SUMMARY_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V3_SUMMARY_PATH}
        ".read ${_v3s_gen_sql}"
    DEPENDS ${_v3s_gen_sql} ${_v3s_schema_src} ${_v3s_data_src}
    COMMENT "Generating synthetic v3 summary test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v3_summary_fixture
    DEPENDS ${ROCPD_DB_V3_SUMMARY_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v3_summary_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V3_SUMMARY_PATH="${ROCPD_DB_V3_SUMMARY_PATH}"
)

# Synthetic v3 missing-metadata naming-fallback fixture: an unnamed stream, a
# region on an absent thread, a region on a NULL-named thread, and an agent with
# a NULL type_index light the synthetic-name / dropped-agent fallbacks.
set(_v3mm_uuid "00000000000000000000000000730048")
set(_v3mm_schema_src
    ${CMAKE_SOURCE_DIR}/source/data_storage/schema/rocpd_tables.sql
)
set(_v3mm_data_src
    ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v3_missing_meta_data.sql
)
set(_v3mm_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v3_missing_meta_gen.sql)
file(READ ${_v3mm_schema_src} _v3mm_schema)
file(READ ${_v3mm_data_src} _v3mm_data)
string(
    REGEX REPLACE
    "INSERT INTO[^;]*rocpd_metadata[^;]*;"
    ""
    _v3mm_schema
    "${_v3mm_schema}"
)
string(REPLACE "{{uuid}}" "_${_v3mm_uuid}" _v3mm_schema "${_v3mm_schema}")
string(REPLACE "{{guid}}" "${_v3mm_uuid}" _v3mm_schema "${_v3mm_schema}")
string(REPLACE "{{uuid}}" "_${_v3mm_uuid}" _v3mm_data "${_v3mm_data}")
string(REPLACE "{{guid}}" "${_v3mm_uuid}" _v3mm_data "${_v3mm_data}")
file(WRITE ${_v3mm_gen_sql} "${_v3mm_schema}\n${_v3mm_data}\n")

set(ROCPD_DB_V3_MISSING_META_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v3_missing_meta.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V3_MISSING_META_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V3_MISSING_META_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V3_MISSING_META_PATH}
        ".read ${_v3mm_gen_sql}"
    DEPENDS ${_v3mm_gen_sql} ${_v3mm_schema_src} ${_v3mm_data_src}
    COMMENT "Generating synthetic v3 missing-metadata test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v3_missing_meta_fixture
    DEPENDS ${ROCPD_DB_V3_MISSING_META_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v3_missing_meta_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V3_MISSING_META_PATH="${ROCPD_DB_V3_MISSING_META_PATH}"
)

set(_v4ap_uuid "0000000000000000000000000a014004")
set(_v4ap_data_src ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4_amb_pmc_data.sql)
set(_v4ap_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v4_amb_pmc_gen.sql)
file(READ ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql _v4ap_schema)
file(READ ${_v4ap_data_src} _v4ap_data)
string(REPLACE "{{uuid}}" "_${_v4ap_uuid}" _v4ap_schema "${_v4ap_schema}")
string(REPLACE "{{guid}}" "${_v4ap_uuid}" _v4ap_schema "${_v4ap_schema}")
string(REPLACE "{{uuid}}" "_${_v4ap_uuid}" _v4ap_data "${_v4ap_data}")
string(REPLACE "{{guid}}" "${_v4ap_uuid}" _v4ap_data "${_v4ap_data}")
file(WRITE ${_v4ap_gen_sql} "${_v4ap_schema}\n${_v4ap_data}\n")

set(ROCPD_DB_V4_AMB_PMC_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v4_amb_pmc.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V4_AMB_PMC_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V4_AMB_PMC_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V4_AMB_PMC_PATH}
        ".read ${_v4ap_gen_sql}"
    DEPENDS
        ${_v4ap_gen_sql}
        ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql
        ${_v4ap_data_src}
    COMMENT "Generating synthetic v4.0 ambiguous-pmc test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v4_amb_pmc_fixture
    DEPENDS ${ROCPD_DB_V4_AMB_PMC_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v4_amb_pmc_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V4_AMB_PMC_PATH="${ROCPD_DB_V4_AMB_PMC_PATH}"
)

set(_v4ac_uuid "0000000000000000000000000a018001")
set(_v4ac_data_src ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4_amb_cls_data.sql)
set(_v4ac_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v4_amb_cls_gen.sql)
file(READ ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql _v4ac_schema)
file(READ ${_v4ac_data_src} _v4ac_data)
string(REPLACE "{{uuid}}" "_${_v4ac_uuid}" _v4ac_schema "${_v4ac_schema}")
string(REPLACE "{{guid}}" "${_v4ac_uuid}" _v4ac_schema "${_v4ac_schema}")
string(REPLACE "{{uuid}}" "_${_v4ac_uuid}" _v4ac_data "${_v4ac_data}")
string(REPLACE "{{guid}}" "${_v4ac_uuid}" _v4ac_data "${_v4ac_data}")
file(WRITE ${_v4ac_gen_sql} "${_v4ac_schema}\n${_v4ac_data}\n")

set(ROCPD_DB_V4_AMB_CLS_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v4_amb_cls.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V4_AMB_CLS_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V4_AMB_CLS_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V4_AMB_CLS_PATH}
        ".read ${_v4ac_gen_sql}"
    DEPENDS
        ${_v4ac_gen_sql}
        ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql
        ${_v4ac_data_src}
    COMMENT "Generating synthetic v4.0 ambiguous-classification test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v4_amb_cls_fixture
    DEPENDS ${ROCPD_DB_V4_AMB_CLS_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v4_amb_cls_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V4_AMB_CLS_PATH="${ROCPD_DB_V4_AMB_CLS_PATH}"
)

# Synthetic v3 ambiguous-pmc fixture: one ambiguous pmc_id (2 rocpd_pmc_event
# rows per event_id) and one clean pmc_id, so get_all_pmc_info() ambiguity
# detection can be asserted.
set(_v3ap_uuid "0000000000000000000000000a014003")
set(_v3ap_schema_src
    ${CMAKE_SOURCE_DIR}/source/data_storage/schema/rocpd_tables.sql
)
set(_v3ap_data_src ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v3_amb_pmc_data.sql)
set(_v3ap_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v3_amb_pmc_gen.sql)
file(READ ${_v3ap_schema_src} _v3ap_schema)
file(READ ${_v3ap_data_src} _v3ap_data)
string(
    REGEX REPLACE
    "INSERT INTO[^;]*rocpd_metadata[^;]*;"
    ""
    _v3ap_schema
    "${_v3ap_schema}"
)
string(REPLACE "{{uuid}}" "_${_v3ap_uuid}" _v3ap_schema "${_v3ap_schema}")
string(REPLACE "{{guid}}" "${_v3ap_uuid}" _v3ap_schema "${_v3ap_schema}")
string(REPLACE "{{uuid}}" "_${_v3ap_uuid}" _v3ap_data "${_v3ap_data}")
string(REPLACE "{{guid}}" "${_v3ap_uuid}" _v3ap_data "${_v3ap_data}")
file(WRITE ${_v3ap_gen_sql} "${_v3ap_schema}\n${_v3ap_data}\n")

set(ROCPD_DB_V3_AMB_PMC_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v3_amb_pmc.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V3_AMB_PMC_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V3_AMB_PMC_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V3_AMB_PMC_PATH}
        ".read ${_v3ap_gen_sql}"
    DEPENDS ${_v3ap_gen_sql} ${_v3ap_schema_src} ${_v3ap_data_src}
    COMMENT "Generating synthetic v3 ambiguous-pmc test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v3_amb_pmc_fixture
    DEPENDS ${ROCPD_DB_V3_AMB_PMC_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v3_amb_pmc_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V3_AMB_PMC_PATH="${ROCPD_DB_V3_AMB_PMC_PATH}"
)

# Synthetic v3 dma-by-destination-agent fixture: 48 memory copies fully cross two
# destination agents and two streams (12 per cell) on one queue, so dma tracks
# keyed by dst_agent_id yield exactly two agent-tracks spanning both streams.
set(_v3da_uuid "00000000000000000000000000d30003")
set(_v3da_schema_src
    ${CMAKE_SOURCE_DIR}/source/data_storage/schema/rocpd_tables.sql
)
set(_v3da_data_src
    ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v3_dma_agent_data.sql
)
set(_v3da_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v3_dma_agent_gen.sql)
file(READ ${_v3da_schema_src} _v3da_schema)
file(READ ${_v3da_data_src} _v3da_data)
string(
    REGEX REPLACE
    "INSERT INTO[^;]*rocpd_metadata[^;]*;"
    ""
    _v3da_schema
    "${_v3da_schema}"
)
string(REPLACE "{{uuid}}" "_${_v3da_uuid}" _v3da_schema "${_v3da_schema}")
string(REPLACE "{{guid}}" "${_v3da_uuid}" _v3da_schema "${_v3da_schema}")
string(REPLACE "{{uuid}}" "_${_v3da_uuid}" _v3da_data "${_v3da_data}")
string(REPLACE "{{guid}}" "${_v3da_uuid}" _v3da_data "${_v3da_data}")
file(WRITE ${_v3da_gen_sql} "${_v3da_schema}\n${_v3da_data}\n")

set(ROCPD_DB_V3_DMA_AGENT_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v3_dma_agent.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V3_DMA_AGENT_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V3_DMA_AGENT_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V3_DMA_AGENT_PATH}
        ".read ${_v3da_gen_sql}"
    DEPENDS ${_v3da_gen_sql} ${_v3da_schema_src} ${_v3da_data_src}
    COMMENT "Generating synthetic v3 dma-by-destination-agent test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v3_dma_agent_fixture
    DEPENDS ${ROCPD_DB_V3_DMA_AGENT_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v3_dma_agent_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V3_DMA_AGENT_PATH="${ROCPD_DB_V3_DMA_AGENT_PATH}"
)

# Synthetic v3 bracket-in-name counter fixture: a counter track whose metric name
# itself contains " [" ("TCC_HIT [sum] [0]") defeats the ordinal strip, so pmc
# resolution falls to the pmc_id tiebreaker.
set(_v3bn_uuid "00000000000000000000000000630063")
set(_v3bn_schema_src
    ${CMAKE_SOURCE_DIR}/source/data_storage/schema/rocpd_tables.sql
)
set(_v3bn_data_src
    ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v3_bracket_name_data.sql
)
set(_v3bn_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v3_bracket_name_gen.sql)
file(READ ${_v3bn_schema_src} _v3bn_schema)
file(READ ${_v3bn_data_src} _v3bn_data)
string(
    REGEX REPLACE
    "INSERT INTO[^;]*rocpd_metadata[^;]*;"
    ""
    _v3bn_schema
    "${_v3bn_schema}"
)
string(REPLACE "{{uuid}}" "_${_v3bn_uuid}" _v3bn_schema "${_v3bn_schema}")
string(REPLACE "{{guid}}" "${_v3bn_uuid}" _v3bn_schema "${_v3bn_schema}")
string(REPLACE "{{uuid}}" "_${_v3bn_uuid}" _v3bn_data "${_v3bn_data}")
string(REPLACE "{{guid}}" "${_v3bn_uuid}" _v3bn_data "${_v3bn_data}")
file(WRITE ${_v3bn_gen_sql} "${_v3bn_schema}\n${_v3bn_data}\n")

set(ROCPD_DB_V3_BRACKET_NAME_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v3_bracket_name.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V3_BRACKET_NAME_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V3_BRACKET_NAME_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V3_BRACKET_NAME_PATH}
        ".read ${_v3bn_gen_sql}"
    DEPENDS ${_v3bn_gen_sql} ${_v3bn_schema_src} ${_v3bn_data_src}
    COMMENT "Generating synthetic v3 bracket-in-name test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v3_bracket_name_fixture
    DEPENDS ${ROCPD_DB_V3_BRACKET_NAME_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v3_bracket_name_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V3_BRACKET_NAME_PATH="${ROCPD_DB_V3_BRACKET_NAME_PATH}"
)

# Real v4.0 capture fixture: rebuilt at build time from a committed,
# un-templated SQL dump of the original captured binary
# (bd54a356ac:tests/unit/rocpd_v4.db). Unlike the synthetic fixtures above,
# this dump carries no {{uuid}}/{{guid}} placeholders -- the literal values
# are the real capture's identity and are read straight into sqlite3.
set(_v4rc_data_src
    ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4_realcapture_data.sql
)

set(ROCPD_DB_V4_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v4.db)
add_custom_command(
    OUTPUT ${ROCPD_DB_V4_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V4_PATH}
    COMMAND ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V4_PATH} ".read ${_v4rc_data_src}"
    DEPENDS ${_v4rc_data_src}
    COMMENT "Generating real v4.0 capture test fixture from committed SQL dump"
    VERBATIM
)
add_custom_target(
    profiler-hub_v4_realcapture_fixture
    DEPENDS ${ROCPD_DB_V4_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v4_realcapture_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PRIVATE ROCPD_DB_V4_PATH="${ROCPD_DB_V4_PATH}"
)

gtest_discover_tests(
    profiler-hub_unit_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES LABELS "unit"
)
