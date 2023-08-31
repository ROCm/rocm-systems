#
#   Tests for the common library
#
project(rocprofiler-tests-common LANGUAGES C CXX)

include(GoogleTest)

set(common_sources demangling.cpp environment.cpp mpl.cpp)

add_executable(common-tests)
target_sources(common-tests PRIVATE ${common_sources})
target_link_libraries(
    common-tests
    PRIVATE rocprofiler::rocprofiler-headers rocprofiler::rocprofiler-common-library
            GTest::gtest GTest::gtest_main)

gtest_add_tests(
    TARGET common-tests
    SOURCES ${common_sources}
    TEST_LIST common-tests_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(${common-tests_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests")
