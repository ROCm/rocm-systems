rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(ROCPROFILER_LIB_PARSER_TEST_SOURCES "parser_test.cpp")

add_executable(parser-test)

target_sources(parser-test PRIVATE ${ROCPROFILER_LIB_PARSER_TEST_SOURCES})

target_link_libraries(
    parser-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
            GTest::gtest_main)

gtest_add_tests(
    TARGET parser-test
    SOURCES ${ROCPROFILER_LIB_PARSER_TEST_SOURCES}
    TEST_LIST parser-tests_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${parser-tests_TESTS}
    PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
