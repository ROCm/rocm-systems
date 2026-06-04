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
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:$ENV{LD_LIBRARY_PATH}"
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
