#
#
#
rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

add_library(aqlprofile-util-tests-flags INTERFACE)
target_compile_options(aqlprofile-util-tests-flags
                       INTERFACE -Wno-shadow -Wno-unused-parameter -Wno-unused-variable)

set(AQLPROFILE_UTIL_TEST_SOURCES util_tests.cpp)

add_executable(aqlprofile-util-tests)

target_sources(aqlprofile-util-tests PRIVATE ${AQLPROFILE_UTIL_TEST_SOURCES})

target_link_libraries(
    aqlprofile-util-tests
    PRIVATE -Wl,--start-group
            rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-aqlprofile
            -Wl,--end-group
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            rocprofiler-sdk::rocprofiler-sdk-drm
            aqlprofile-util-tests-flags
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main)

# Ensure hsa runtime is in DT_NEEDED
target_link_options(aqlprofile-util-tests PRIVATE -Wl,--no-as-needed
                    $<TARGET_FILE:hsa-runtime64::hsa-runtime64> -Wl,--as-needed)

rocprofiler_add_unit_test(
    TARGET aqlprofile-util-tests
    SOURCES ${AQLPROFILE_UTIL_TEST_SOURCES}
    TEST_PREFIX "aqlprofile.util."
    TEST_LIST aqlprofile_util_TESTS
    TIMEOUT 30
    LABELS "unittests;aqlprofile"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
