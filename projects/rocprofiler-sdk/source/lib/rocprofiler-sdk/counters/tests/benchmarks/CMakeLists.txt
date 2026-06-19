rocprofiler_deactivate_clang_tidy()

project(rocprofiler-sdk-replay-benchmarks LANGUAGES CXX HIP)

include(GoogleTest)

add_executable(replay-benchmarks)

target_sources(replay-benchmarks PRIVATE replay_benchmarks.cpp)

target_link_libraries(
    replay-benchmarks
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-hip
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library
            GTest::gtest
            GTest::gtest_main)

set(_ROCPROFILER_SHARE_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}")

rocprofiler_add_unit_test(
    TARGET replay-benchmarks
    SOURCES replay_benchmarks.cpp
    TIMEOUT 120
    ENVIRONMENT
        "ROCPROFILER_METRICS_PATH=${_ROCPROFILER_SHARE_DIR}"
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
