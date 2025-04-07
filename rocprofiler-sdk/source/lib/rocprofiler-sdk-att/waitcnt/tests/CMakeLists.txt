#
# Waitcnt ISA tests
#

add_executable(att-decoder-waitcnt-test)
target_link_libraries(
    att-decoder-waitcnt-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-att-parser
            rocprofiler-sdk::rocprofiler-sdk-json
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-glog
            rocprofiler-sdk::rocprofiler-sdk-static-library
            GTest::gtest
            GTest::gtest_main)
target_sources(att-decoder-waitcnt-test PRIVATE att_decoder_waitcnt_test.cpp)

gtest_add_tests(
    TARGET att-decoder-waitcnt-test
    SOURCES att_decoder_waitcnt_test.cpp
    TEST_LIST att-decoder-waitcnt-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(${att-decoder-waitcnt-test_TESTS} PROPERTIES TIMEOUT 10 LABELS
                                                                  "unittests")
