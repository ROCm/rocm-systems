# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(demangler-tests OBJECT demangler_test.cpp)

target_link_libraries(
    demangler-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(demangler-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(track-registry-tests OBJECT test_track_registry.cpp)

target_link_libraries(
    track-registry-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(track-registry-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(packet-framing-tests OBJECT test_packet_framing.cpp)

target_link_libraries(
    packet-framing-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(packet-framing-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(perfetto-engine-tests OBJECT test_perfetto_engine.cpp)

add_library(perfetto-output-layout-tests OBJECT test_perfetto_output_layout.cpp)

target_link_libraries(
    perfetto-output-layout-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(
    perfetto-output-layout-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib
)

target_link_libraries(
    perfetto-engine-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
        rocprofiler-systems-headers
        rocprofiler-systems-timemory
)

target_include_directories(perfetto-engine-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(perfetto-sinks-tests OBJECT test_perfetto_sinks.cpp)

target_link_libraries(
    perfetto-sinks-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(perfetto-sinks-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(
    emission-handshake-tls-contract-tests
    OBJECT
    test_emission_handshake_tls_contract.cpp
)

target_link_libraries(
    emission-handshake-tls-contract-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(
    emission-handshake-tls-contract-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib
)

add_library(output-format-tests OBJECT test_output_format.cpp)

target_link_libraries(
    output-format-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
        rocprofiler-systems-headers
        rocprofiler-systems-timemory
)

target_include_directories(
    output-format-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib/
)
