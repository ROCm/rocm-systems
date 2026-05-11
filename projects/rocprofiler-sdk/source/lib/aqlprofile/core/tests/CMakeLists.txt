#
#
#
rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

add_library(aqlprofile-core-tests-flags INTERFACE)
target_compile_options(aqlprofile-core-tests-flags
                       INTERFACE -Wno-shadow -Wno-unused-parameter -Wno-unused-variable)

set(AQLPROFILE_CORE_TEST_SOURCES
    aql_profile_tests.cpp aql_profile_v2_tests.cpp command_buffer_tests.cpp
    counter_tests.cpp logger_tests.cpp memorymanager_tests.cpp pm4_factory_tests.cpp
    aql_profile_v2_c_test.c)

add_executable(aqlprofile-core-tests)

target_sources(aqlprofile-core-tests PRIVATE ${AQLPROFILE_CORE_TEST_SOURCES})
set_target_properties(aqlprofile-core-tests PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED
                                                                     ON)

target_link_libraries(
    aqlprofile-core-tests
    PRIVATE -Wl,--start-group
            rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-aqlprofile
            -Wl,--end-group
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            rocprofiler-sdk::rocprofiler-sdk-drm
            aqlprofile-core-tests-flags
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main)

rocprofiler_add_unit_test(
    TARGET aqlprofile-core-tests
    SOURCES ${AQLPROFILE_CORE_TEST_SOURCES}
    TEST_PREFIX "aqlprofile.core."
    TEST_LIST aqlprofile_core_TESTS
    TIMEOUT 30
    LABELS "unittests;aqlprofile"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
