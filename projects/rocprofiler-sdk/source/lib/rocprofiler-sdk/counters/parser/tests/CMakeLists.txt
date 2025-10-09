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

rocprofiler_add_unit_test(
    TARGET parser-test
    SOURCES ${ROCPROFILER_LIB_PARSER_TEST_SOURCES}
    TIMEOUT 45
    LABELS "unittests"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
