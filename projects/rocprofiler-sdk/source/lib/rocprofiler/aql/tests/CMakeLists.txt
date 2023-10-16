rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(ROCPROFILER_LIB_AQL_TEST_SOURCES "aql_test.cpp")

add_executable(aql-test)

target_sources(aql-test PRIVATE ${ROCPROFILER_LIB_AQL_TEST_SOURCES})

target_link_libraries(
    aql-test
    PRIVATE rocprofiler::rocprofiler-static-library rocprofiler::rocprofiler-glog
            rocprofiler::rocprofiler-hip rocprofiler::rocprofiler-common-library
            GTest::gtest GTest::gtest_main)

gtest_add_tests(
    TARGET aql-test
    SOURCES ${ROCPROFILER_LIB_AQL_TEST_SOURCES}
    TEST_LIST aql-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(${aql-test_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests")
