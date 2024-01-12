rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(ROCPROFILER_LIB_COUNTER_TEST_SOURCES metrics_test.cpp evaluate_ast_test.cpp
                                         dimension.cpp init_order.cpp)

add_executable(counter-test)

target_sources(counter-test PRIVATE ${ROCPROFILER_LIB_COUNTER_TEST_SOURCES})

target_link_libraries(
    counter-test
    PRIVATE rocprofiler::rocprofiler-hip rocprofiler::rocprofiler-common-library
            rocprofiler::rocprofiler-static-library GTest::gtest GTest::gtest_main)

gtest_add_tests(
    TARGET counter-test
    SOURCES ${ROCPROFILER_LIB_COUNTER_TEST_SOURCES}
    TEST_LIST counter-tests_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(${counter-tests_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests")
