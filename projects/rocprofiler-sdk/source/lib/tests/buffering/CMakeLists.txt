#
#
#
project(rocprofiler-sdk-tests-buffering LANGUAGES C CXX)

include(GoogleTest)

set(buffering_sources buffering-serial.cpp buffering-parallel.cpp buffering-save-load.cpp)

add_executable(buffering-test)
target_sources(buffering-test PRIVATE ${buffering_sources})
target_link_libraries(
    buffering-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-common-library GTest::gtest
            GTest::gtest_main)

gtest_add_tests(
    TARGET buffering-test
    SOURCES ${buffering_sources}
    TEST_LIST buffering-tests_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${buffering-tests_TESTS}
    PROPERTIES TIMEOUT 360 LABELS "unittests" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
