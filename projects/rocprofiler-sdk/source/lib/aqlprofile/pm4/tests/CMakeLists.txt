#
#
#
rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

add_library(aqlprofile-pm4-tests-flags INTERFACE)
target_compile_options(aqlprofile-pm4-tests-flags
                       INTERFACE -Wno-shadow -Wno-unused-parameter -Wno-unused-variable)

set(AQLPROFILE_PM4_TEST_SOURCES
    cmd_builder_tests.cpp gfx9_cmd_builder_tests.cpp pmc_builder_tests.cpp
    spm_builder_test.cpp sqtt_builder_tests.cpp trace_config_test.cpp)

add_executable(aqlprofile-pm4-tests)

target_sources(aqlprofile-pm4-tests PRIVATE ${AQLPROFILE_PM4_TEST_SOURCES})

target_link_libraries(
    aqlprofile-pm4-tests
    PRIVATE -Wl,--start-group
            rocprofiler-sdk::rocprofiler-sdk-aqlprofile
            rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-common-library
            -Wl,--end-group
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            rocprofiler-sdk::rocprofiler-sdk-drm
            aqlprofile-pm4-tests-flags
            GTest::gtest
            GTest::gmock
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET aqlprofile-pm4-tests
    SOURCES ${AQLPROFILE_PM4_TEST_SOURCES}
    TEST_PREFIX "aqlprofile.pm4."
    TEST_LIST aqlprofile_pm4_TESTS DISABLE_TESTS "SqttBuilderTest.BufferStepCalculation"
    TIMEOUT 30
    LABELS "unittests;aqlprofile"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
