#
#
#
rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(ROCPROFILER_LIB_PLATFORM_TEST_SOURCES platform_dispatch.cpp)

add_executable(platform-dispatch-tests)
target_sources(platform-dispatch-tests PRIVATE ${ROCPROFILER_LIB_PLATFORM_TEST_SOURCES})
target_link_libraries(
    platform-dispatch-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-common-library GTest::gtest
            GTest::gtest_main)

if(WIN32)
    # Windows resolves dependent DLLs via PATH (no LD_LIBRARY_PATH). The sysfs/KFD
    # topology fixture is Linux-only, so the Windows tests run without it - they exercise
    # the D3DKMT negative path, which needs no fixture.
    rocprofiler_add_unit_test(
        TARGET platform-dispatch-tests
        SOURCES ${ROCPROFILER_LIB_PLATFORM_TEST_SOURCES}
        TIMEOUT 30
        LABELS "unittests;platform-dispatch"
        ENVIRONMENT "PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}\\;$ENV{PATH}"
        FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
else()
    # Reuses the sysfs topology fixture from lib/tests so the gnulinux force-platform test
    # can resolve a real node tree on bare-metal CI. The sysfs topology should already
    # exist in the build directory so do not copy over the data again, especially since
    # rocprofiler_add_unit_test does not support DATA / CONFIGURE_FILES in a non-relative
    # child directory path.
    rocprofiler_add_unit_test(
        TARGET platform-dispatch-tests
        SOURCES ${ROCPROFILER_LIB_PLATFORM_TEST_SOURCES}
        TIMEOUT 30
        LABELS "unittests;platform-dispatch"
        ENVIRONMENT
            "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:$ENV{LD_LIBRARY_PATH}"
        FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endif()
