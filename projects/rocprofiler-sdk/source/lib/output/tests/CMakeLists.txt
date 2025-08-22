#
#   Tests for the output library
#
project(rocprofiler-sdk-tests-output-lib LANGUAGES C CXX)

include(GoogleTest)

set(output_lib_test_sources output_key.cpp)

add_executable(output-lib-tests)
target_sources(output-lib-tests PRIVATE ${output_lib_test_sources})
target_link_libraries(
    output-lib-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-output-library
            rocprofiler-sdk::rocprofiler-sdk-cereal
            GTest::gtest
            GTest::gtest_main)

gtest_add_tests(
    TARGET output-lib-tests
    SOURCES ${lib_output_test_sources}
    TEST_LIST output-lib-tests_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${output-lib-tests_TESTS}
    PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}" ENVIRONMENT "TEST_LOG_LEVEL=info")
