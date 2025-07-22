#
#   Tests for the common library
#
project(rocprofiler-sdk-tests-common LANGUAGES C CXX)

include(GoogleTest)

set(common_sources c_array.cpp demangling.cpp environment.cpp md5sum.cpp mpl.cpp
                   parse.cpp sha256.cpp uuid_v7.cpp)

add_executable(common-tests)
target_sources(common-tests PRIVATE ${common_sources})
target_link_libraries(
    common-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-output-library
            rocprofiler-sdk::rocprofiler-sdk-cereal
            GTest::gtest
            GTest::gtest_main)

gtest_add_tests(
    TARGET common-tests
    SOURCES ${common_sources}
    TEST_LIST common-tests_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${common-tests_TESTS}
    PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}" ENVIRONMENT "TEST_LOG_LEVEL=info")
