#
#
#

rocprofiler_deactivate_clang_tidy()

add_executable(att-parser-tool-v3)
target_sources(att-parser-tool-v3 PRIVATE standalone_tool_main.cpp)
target_link_libraries(
    att-parser-tool-v3
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-att-parser
            rocprofiler-sdk::rocprofiler-sdk-json
            rocprofiler-sdk::rocprofiler-sdk-common-library)

add_executable(att-decoder-test)
target_sources(att-decoder-test PRIVATE att_decoder_test.cpp)
target_link_libraries(
    att-decoder-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-att-parser
            rocprofiler-sdk::rocprofiler-sdk-json
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-glog
            rocprofiler-sdk::rocprofiler-sdk-static-library
            GTest::gtest
            GTest::gtest_main)

add_library(att_decoder_testing SHARED)
add_library(rocprofiler-sdk::att-decoder-testing ALIAS att_decoder_testing)
target_sources(att_decoder_testing PRIVATE dummy_decoder.cpp)
set_target_properties(att_decoder_testing PROPERTIES VERSION ${PROJECT_VERSION}
                                                     SOVERSION ${PROJECT_VERSION_MAJOR})

add_library(att_decoder_summary SHARED)
add_library(rocprofiler-sdk::att-decoder-summary ALIAS att_decoder_summary)
target_sources(att_decoder_summary PRIVATE dummy_decoder.cpp)
set_target_properties(
    att_decoder_summary
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/att
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR})

if(NOT ROCPROFILER_MEMCHECK)
    set(IS_MEMCHECK OFF)
else()
    set(IS_MEMCHECK ON)
endif()

set(env-att-lib "ROCPROF_ATT_LIBRARY_PATH=${PROJECT_BINARY_DIR}/lib")
gtest_add_tests(
    TARGET att-decoder-test
    SOURCES att_decoder_test.cpp
    TEST_LIST att-decoder-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(
    ${att-decoder-test_TESTS} PROPERTIES ENVIRONMENT "${env-att-lib}" TIMEOUT 10 LABELS
                                         "unittests" DISABLED "${IS_MEMCHECK}")
