# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(packet-framing-tests OBJECT test_packet_framing.cpp)

target_link_libraries(
    packet-framing-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(packet-framing-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(perfetto-engine-tests OBJECT test_perfetto_engine.cpp)

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
