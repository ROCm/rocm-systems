rocprofiler_deactivate_clang_tidy()

project(rocprofiler-sdk-unit-tests-aql LANGUAGES C CXX)

include(GoogleTest)

set(ROCPROFILER_LIB_AQL_TEST_SOURCES aql_test.cpp helpers.cpp)

add_executable(aql-test)
target_sources(aql-test PRIVATE ${ROCPROFILER_LIB_AQL_TEST_SOURCES})
target_link_libraries(
    aql-test
    PRIVATE rocprofiler-sdk::counter-test-constants
            rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-glog
            rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-hip
            rocprofiler-sdk::rocprofiler-sdk-common-library
            GTest::gtest
            GTest::gtest_main)

set(_ROCPROFILER_SHARE_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}")

rocprofiler_add_unit_test(
    TARGET aql-test
    SOURCES ${ROCPROFILER_LIB_AQL_TEST_SOURCES}
    TIMEOUT 45
    ENVIRONMENT
        "ROCPROFILER_METRICS_PATH=${_ROCPROFILER_SHARE_DIR}"
        "LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:/opt/rocm/lib:/opt/rocm/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
