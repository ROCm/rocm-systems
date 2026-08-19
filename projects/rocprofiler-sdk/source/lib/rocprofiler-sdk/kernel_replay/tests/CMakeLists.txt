rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

# Exercises only the configuration path via rocprofiler_force_configure (no GPU / HSA).
add_executable(kernel-replay-configure-test)
target_sources(kernel-replay-configure-test PRIVATE replay_configure.cpp)

target_link_libraries(
    kernel-replay-configure-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET kernel-replay-configure-test
    SOURCES replay_configure.cpp
    TIMEOUT 60
    ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
