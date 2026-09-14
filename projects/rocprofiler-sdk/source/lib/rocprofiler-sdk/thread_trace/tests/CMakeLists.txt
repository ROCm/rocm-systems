rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(ROCPROFILER_THREAD_TRACE_PACKET_TEST_SOURCES "att_packet_test.cpp")

add_executable(thread-trace-packet-test)

target_sources(thread-trace-packet-test
               PRIVATE ${ROCPROFILER_THREAD_TRACE_PACKET_TEST_SOURCES})

target_link_libraries(
    thread-trace-packet-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-abseil
            rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-hip
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            GTest::gtest
            GTest::gtest_main
            rocprofiler-sdk::counter-test-constants)

set(_ROCPROFILER_SHARE_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}")

rocprofiler_add_unit_test(
    TARGET thread-trace-packet-test
    SOURCES ${ROCPROFILER_THREAD_TRACE_PACKET_TEST_SOURCES}
    TEST_LIST thread-trace-test_TESTS
    TIMEOUT 10
    LABELS "unittests"
    ENVIRONMENT
        "ROCPROFILER_METRICS_PATH=${_ROCPROFILER_SHARE_DIR}"
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

set(ROCPROFILER_THREAD_TRACE_PRODUCER_CONSUMER_TEST_SOURCES "producer_consumer.cpp")

add_executable(thread-trace-producer-consumer-test)
target_sources(thread-trace-producer-consumer-test
               PRIVATE ${ROCPROFILER_THREAD_TRACE_PRODUCER_CONSUMER_TEST_SOURCES})

target_link_libraries(
    thread-trace-producer-consumer-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-abseil
            rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-hip
            rocprofiler-sdk::rocprofiler-sdk-common-library
            GTest::gtest
            GTest::gtest_main
            rocprofiler-sdk::counter-test-constants)

# The producer/consumer harness exercises the AQL profile thread-trace path with triple
# buffering enabled, which is not supported by aqlprofile on GFX11
# (aqlprofile_att_create_packets() throws "Not supported"). Reuse the shared SQTT
# triple-buffer architecture gate to skip this suite on unsupported GPUs.
include(rocprofiler-sdk-utilities)
rocprofiler_sdk_sqtt_triple_buffer_disabled(THREAD_TRACE_PC_TEST_DISABLED)

rocprofiler_add_unit_test(
    TARGET thread-trace-producer-consumer-test
    SOURCES ${ROCPROFILER_THREAD_TRACE_PRODUCER_CONSUMER_TEST_SOURCES}
    TEST_LIST thread-trace-test_TESTS
    TIMEOUT 100
    LABELS "unittests"
    DISABLED ${THREAD_TRACE_PC_TEST_DISABLED}
    ENVIRONMENT
        "ROCPROFILER_METRICS_PATH=${_ROCPROFILER_SHARE_DIR}"
        "LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:/opt/rocm/lib:/opt/rocm/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Stand-in decoders for the analysis test. "v2" offers the handle-based API that analyses
# need, "v1" offers only what a decoder older than 0.2.3 had. DL appends the decoder's
# real filename to a directory, so each variant needs a directory of its own.
foreach(_FAKE_DECODER_VARIANT "v1" "v2")
    set(_FAKE_DECODER_TARGET fake-trace-decoder-${_FAKE_DECODER_VARIANT})

    add_library(${_FAKE_DECODER_TARGET} SHARED)
    target_sources(${_FAKE_DECODER_TARGET} PRIVATE fake_trace_decoder.cpp
                                                   fake_trace_decoder.hpp)
    target_include_directories(${_FAKE_DECODER_TARGET}
                               PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)
    target_link_libraries(
        ${_FAKE_DECODER_TARGET}
        PRIVATE $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-headers>
                $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-build-flags>)
    set_target_properties(
        ${_FAKE_DECODER_TARGET}
        PROPERTIES OUTPUT_NAME rocprof-trace-decoder
                   LIBRARY_OUTPUT_DIRECTORY
                   ${CMAKE_CURRENT_BINARY_DIR}/${_FAKE_DECODER_VARIANT})
endforeach()

target_compile_definitions(fake-trace-decoder-v2
                           PRIVATE FAKE_TRACE_DECODER_SUPPORTS_ANALYSIS=1)

set(ROCPROFILER_THREAD_TRACE_DECODER_ANALYSIS_TEST_SOURCES "decoder_analysis_test.cpp")

add_executable(thread-trace-decoder-analysis-test)
target_sources(thread-trace-decoder-analysis-test
               PRIVATE ${ROCPROFILER_THREAD_TRACE_DECODER_ANALYSIS_TEST_SOURCES})
add_dependencies(thread-trace-decoder-analysis-test fake-trace-decoder-v1
                 fake-trace-decoder-v2)

target_compile_definitions(
    thread-trace-decoder-analysis-test
    PRIVATE FAKE_TRACE_DECODER_V1_DIR="${CMAKE_CURRENT_BINARY_DIR}/v1"
            FAKE_TRACE_DECODER_V2_DIR="${CMAKE_CURRENT_BINARY_DIR}/v2")

target_link_libraries(
    thread-trace-decoder-analysis-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-abseil
            rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-hip
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET thread-trace-decoder-analysis-test
    SOURCES ${ROCPROFILER_THREAD_TRACE_DECODER_ANALYSIS_TEST_SOURCES}
    TEST_LIST thread-trace-test_TESTS
    TIMEOUT 10
    LABELS "unittests"
    ENVIRONMENT
        "ROCPROFILER_METRICS_PATH=${_ROCPROFILER_SHARE_DIR}"
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
