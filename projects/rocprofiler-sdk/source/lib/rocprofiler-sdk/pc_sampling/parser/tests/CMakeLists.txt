rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_TEST_SOURCES pcs_parser.cpp)
set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_TEST_HEADERS mocks.hpp)

add_executable(pcs-parser-test)

target_sources(pcs-parser-test PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_TEST_SOURCES}
                                       ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_TEST_HEADERS})
# $<TARGET_OBJECTS:rocprofiler::rocprofiler-object-library>)

target_link_libraries(
    pcs-parser-test
    PRIVATE rocprofiler::rocprofiler-common-library
            rocprofiler::rocprofiler-static-library GTest::gtest GTest::gtest_main)

gtest_add_tests(
    TARGET pcs-parser-test
    SOURCES ${ROCPROFILER_LIB_COUNTER_TEST_SOURCES}
    TEST_LIST pcs-parser-tests_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(${pcs-parser-tests_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests")
