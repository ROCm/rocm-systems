#
#
#
rocprofiler_deactivate_clang_tidy()

include(GoogleTest)
include(rocprofiler-sdk-utilities)

set(ROCPROFILER_LIB_PLATFORM_TEST_SOURCES platform_dispatch.cpp)
set(ROCPROFILER_LIB_PLATFORM_TEST_HELPERS)

rocprofiler_sdk_dxg_topology_supported(ROCPROFILER_SDK_DXG_TOPOLOGY_SUPPORTED)

if(ROCPROFILER_SDK_DXG_TOPOLOGY_SUPPORTED)
    list(APPEND ROCPROFILER_LIB_PLATFORM_TEST_SOURCES wsl_dxg_topology.cpp
         wsl_dxg_thunk.cpp wsl_dxg_loader.cpp)

    # Not linked against the HSA runtime: a real libhsa-runtime64 mapped into the probe
    # would be one more directory for the lookup under test to consider.
    add_executable(wsl-dxg-loader-probe wsl_dxg_loader_probe.cpp)
    target_link_libraries(
        wsl-dxg-loader-probe PRIVATE rocprofiler-sdk::rocprofiler-sdk-static-library
                                     rocprofiler-sdk::rocprofiler-sdk-common-library)

    # A versioned soname of its own and -z nodelete, like librocdxg
    # (libhsakmt/CMakeLists_wsl.txt). The probe asks for this soname wherever the lookup
    # asks for librocdxg's.
    add_library(wsl-dxg-fixture-thunk SHARED wsl_dxg_loader_thunk_fixture.cpp)
    target_link_libraries(wsl-dxg-fixture-thunk
                          PRIVATE rocprofiler-sdk::rocprofiler-sdk-hsakmt-nolink)
    target_link_options(wsl-dxg-fixture-thunk PRIVATE "LINKER:-z,nodelete")
    set_target_properties(wsl-dxg-fixture-thunk PROPERTIES OUTPUT_NAME rocdxg-fixture
                                                           SOVERSION 0)

    foreach(_KIND runpath rpath plain)
        add_library(wsl-dxg-fixture-${_KIND}-caller SHARED
                    wsl_dxg_loader_caller_fixture.cpp)
        target_link_libraries(wsl-dxg-fixture-${_KIND}-caller PRIVATE ${CMAKE_DL_LIBS})
        # A sibling call would leave dlopen resolving against whatever called the fixture.
        target_compile_options(wsl-dxg-fixture-${_KIND}-caller
                               PRIVATE -fno-optimize-sibling-calls)
    endforeach()
    target_link_options(wsl-dxg-fixture-runpath-caller PRIVATE
                        "LINKER:--enable-new-dtags")
    target_link_options(wsl-dxg-fixture-rpath-caller PRIVATE "LINKER:--disable-new-dtags")
    set_target_properties(wsl-dxg-fixture-runpath-caller wsl-dxg-fixture-rpath-caller
                          PROPERTIES INSTALL_RPATH "\$ORIGIN")
    set_target_properties(wsl-dxg-fixture-thunk wsl-dxg-fixture-plain-caller
                          PROPERTIES INSTALL_RPATH "")

    # Each fixture's run path is part of what it stands for, so the build tree gets the
    # same one as the install tree, and nothing is added from the link path.
    set_target_properties(
        wsl-dxg-fixture-thunk wsl-dxg-fixture-runpath-caller wsl-dxg-fixture-rpath-caller
        wsl-dxg-fixture-plain-caller PROPERTIES BUILD_WITH_INSTALL_RPATH ON
                                                INSTALL_RPATH_USE_LINK_PATH OFF)

    set(ROCPROFILER_LIB_PLATFORM_TEST_HELPERS
        wsl-dxg-loader-probe wsl-dxg-fixture-thunk wsl-dxg-fixture-runpath-caller
        wsl-dxg-fixture-rpath-caller wsl-dxg-fixture-plain-caller)
endif()

add_executable(platform-dispatch-tests)
target_sources(platform-dispatch-tests PRIVATE ${ROCPROFILER_LIB_PLATFORM_TEST_SOURCES})
target_link_libraries(
    platform-dispatch-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-drm
            GTest::gtest
            GTest::gtest_main)

if(ROCPROFILER_SDK_DXG_TOPOLOGY_SUPPORTED)
    foreach(_TARGET platform-dispatch-tests wsl-dxg-loader-probe)
        target_compile_definitions(
            ${_TARGET}
            PRIVATE
                "WSL_DXG_FIXTURE_THUNK=\"$<TARGET_SONAME_FILE_NAME:wsl-dxg-fixture-thunk>\""
            )
    endforeach()
endif()

# Reuses the sysfs topology fixture from lib/tests so the gnulinux force-platform test can
# resolve a real node tree on bare-metal CI. The sysfs topology should already exist in
# the build directory so do not copy over the data again, especially since
# rocprofiler_add_unit_test does not support DATA / CONFIGURE_FILES in a non-relative
# child directory path.
rocprofiler_add_unit_test(
    TARGET platform-dispatch-tests
    SOURCES ${ROCPROFILER_LIB_PLATFORM_TEST_SOURCES}
    HELPERS ${ROCPROFILER_LIB_PLATFORM_TEST_HELPERS}
    TIMEOUT 30
    LABELS "unittests;platform-dispatch"
    ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
