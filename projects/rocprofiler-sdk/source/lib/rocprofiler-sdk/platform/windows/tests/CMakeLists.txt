#
# Unit tests for the native-Windows D3DKMT + wkmi agent enumerator.
#
rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(ROCPROFILER_LIB_PLATFORM_WINDOWS_TEST_SOURCES agent.cpp)

add_executable(platform-windows-agent-tests)
target_sources(platform-windows-agent-tests
               PRIVATE ${ROCPROFILER_LIB_PLATFORM_WINDOWS_TEST_SOURCES})
target_link_libraries(
    platform-windows-agent-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-wkmi GTest::gtest GTest::gtest_main)

# Windows uses PATH (not LD_LIBRARY_PATH) for the DLL search at load time.
rocprofiler_add_unit_test(
    TARGET platform-windows-agent-tests
    SOURCES ${ROCPROFILER_LIB_PLATFORM_WINDOWS_TEST_SOURCES}
    TIMEOUT 30
    LABELS "unittests;platform-windows"
    ENVIRONMENT "PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}\\;$ENV{PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
