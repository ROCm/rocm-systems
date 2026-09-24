rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(ROCPROFILER_LIB_PC_SAMPLING_IOCTL_TEST_SOURCES pc_sampling_versioning.cpp)

add_executable(pcs-ioctl-test)

target_sources(pcs-ioctl-test PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_IOCTL_TEST_SOURCES})

target_link_libraries(
    pcs-ioctl-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET pcs-ioctl-test
    SOURCES ${ROCPROFILER_LIB_PC_SAMPLING_IOCTL_TEST_SOURCES}
    TIMEOUT 45
    LABELS "unittests;pc-sampling;ioctl"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
