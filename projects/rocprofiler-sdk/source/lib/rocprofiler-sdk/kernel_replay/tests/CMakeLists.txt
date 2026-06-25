rocprofiler_deactivate_clang_tidy()

project(rocprofiler-sdk-kernel-replay-tests LANGUAGES CXX)

include(GoogleTest)

set(KERNEL_REPLAY_SNAP_RESTORE_SOURCES snap_restore.cpp)

add_executable(kernel-replay-snapshot-test)
target_sources(kernel-replay-snapshot-test PRIVATE ${KERNEL_REPLAY_SNAP_RESTORE_SOURCES})

target_link_libraries(
    kernel-replay-snapshot-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::counter-test-constants
            GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET kernel-replay-snapshot-test
    SOURCES ${KERNEL_REPLAY_SNAP_RESTORE_SOURCES}
    TIMEOUT 45
    ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
