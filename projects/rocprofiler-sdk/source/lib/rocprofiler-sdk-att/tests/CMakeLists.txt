#
#
#
project(rocprofiler-att-parser-tests LANGUAGES CXX)

add_executable(att-parser-tool-v3)
target_link_libraries(
    att-parser-tool-v3
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-att-parser
            rocprofiler-sdk::rocprofiler-sdk-json
            rocprofiler-sdk::rocprofiler-sdk-common-library)
target_sources(att-parser-tool-v3 PRIVATE standalone_tool_main.cpp)

add_executable(att-decoder-test)
target_link_libraries(
    att-decoder-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-att-parser
            rocprofiler-sdk::rocprofiler-sdk-json
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-glog
            rocprofiler-sdk::rocprofiler-sdk-static-library
            GTest::gtest
            GTest::gtest_main)
target_sources(att-decoder-test PRIVATE att_decoder_test.cpp)

set(env-att-lib "ROCPROF_ATT_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/../lib")
add_library(att_decoder_testing SHARED)
target_sources(att_decoder_testing PRIVATE dummy_decoder.cpp)

if(NOT ROCPROFILER_MEMCHECK)
    gtest_add_tests(
        TARGET att-decoder-test
        SOURCES att_decoder_test.cpp
        TEST_LIST att-decoder-test_TESTS
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(
        ${att-decoder-test_TESTS} PROPERTIES ENVIRONMENT "${env-att-lib}" TIMEOUT 10
                                             LABELS "unittests")
endif()
