rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(ROCPROFILER_THREAD_TRACE_TEST_SOURCES "att_packet_test.cpp")

add_executable(thread-trace-packet-test)

target_sources(thread-trace-packet-test PRIVATE ${ROCPROFILER_THREAD_TRACE_TEST_SOURCES})

target_link_libraries(
    thread-trace-packet-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-glog
            rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-hip
            rocprofiler-sdk::rocprofiler-sdk-common-library
            GTest::gtest
            GTest::gtest_main
            rocprofiler-sdk::counter-test-constants)

gtest_add_tests(
    TARGET thread-trace-packet-test
    SOURCES ${ROCPROFILER_THREAD_TRACE_TEST_SOURCES}
    TEST_LIST thread-trace-packet-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${thread-trace-packet-test_TESTS}
    PROPERTIES TIMEOUT 10 LABELS "unittests" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
