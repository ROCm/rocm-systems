rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(PCTEST_INCLUDE_DIR
    ${PROJECT_SOURCE_DIR}/source/lib/rocprofiler-sdk/pc_sampling/parser/)

set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_TEST_HEADERS mocks.hpp)
set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_TEST_SOURCES mocks.cpp)

set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_ID_TEST_SOURCES
    ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_TEST_SOURCES} correlation_id_test.cpp)
set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_BENCH_TEST_SOURCES
    ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_TEST_SOURCES} benchmark_test.cpp)
set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_GFX9_TEST_SOURCES
    ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_TEST_SOURCES} gfx9test.hpp gfx9test.cpp
    gfx950test.cpp)
set(ROCPROFILER_LIB_PC_SAMPLING_PARSER_MULTIGPU_TEST_SOURCES
    ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_TEST_SOURCES} multigpu.cpp)

add_executable(pcs_gfx9_test)

target_sources(pcs_gfx9_test
               PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_GFX9_TEST_SOURCES})
target_include_directories(pcs_gfx9_test PRIVATE ${PCTEST_INCLUDE_DIR})

target_link_libraries(
    pcs_gfx9_test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
            GTest::gtest_main)

gtest_add_tests(
    TARGET pcs_gfx9_test
    SOURCES ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_GFX9_TEST_SOURCES}
    TEST_LIST pcs_gfx9_test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${pcs_gfx9_test_TESTS}
    PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_executable(pcs_id_test)

target_sources(pcs_id_test PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_ID_TEST_SOURCES})
target_include_directories(pcs_id_test PRIVATE ${PCTEST_INCLUDE_DIR})

target_link_libraries(
    pcs_id_test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
            GTest::gtest_main)

gtest_add_tests(
    TARGET pcs_id_test
    SOURCES ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_ID_TEST_SOURCES}
    TEST_LIST pcs_id_test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${pcs_id_test_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
                                    "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_executable(pcs_bench_test)

target_compile_options(pcs_bench_test PRIVATE "-O3 -ffast-math")
target_sources(pcs_bench_test
               PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_BENCH_TEST_SOURCES})
target_include_directories(pcs_bench_test PRIVATE ${PCTEST_INCLUDE_DIR})

target_link_libraries(
    pcs_bench_test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
            GTest::gtest_main)

add_executable(pcs_thread_test)
target_compile_options(pcs_thread_test PRIVATE "-O3 -ffast-math")

target_sources(pcs_thread_test
               PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_MULTIGPU_TEST_SOURCES})
target_include_directories(pcs_thread_test PRIVATE ${PCTEST_INCLUDE_DIR})

target_link_libraries(
    pcs_thread_test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
            GTest::gtest_main)

gtest_add_tests(
    TARGET pcs_thread_test
    SOURCES ${ROCPROFILER_LIB_PC_SAMPLING_PARSER_MULTIGPU_TEST_SOURCES}
    TEST_LIST pcs_thread_test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${pcs_thread_test_TESTS}
    PROPERTIES TIMEOUT 75 LABELS "unittests" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
