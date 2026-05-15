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

set(_ROCPROFILER_SHARE_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}")

rocprofiler_add_unit_test(
    TARGET parser-test
    SOURCES ${ROCPROFILER_LIB_PARSER_TEST_SOURCES}
    TIMEOUT 45
    LABELS unittests
    ENVIRONMENT
        "ROCPROFILER_METRICS_PATH=${_ROCPROFILER_SHARE_DIR}"
        "LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:/opt/rocm/lib:/opt/rocm/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
