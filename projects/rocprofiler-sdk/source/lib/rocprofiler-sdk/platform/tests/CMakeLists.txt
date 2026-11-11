#
#
#
rocprofiler_deactivate_clang_tidy()

include(GoogleTest)
include(rocprofiler-sdk-utilities)

set(ROCPROFILER_LIB_PLATFORM_TEST_SOURCES platform_dispatch.cpp)

rocprofiler_sdk_dxg_topology_supported(ROCPROFILER_SDK_DXG_TOPOLOGY_SUPPORTED)

# The real-loader cases in wsl_dxg_loader.cpp need a stand-in librocdxg whose DT_SONAME is
# the versioned name the SDK looks for, and only rocdxg_version.h knows its major.
set(_ROCDXG_VERSION_MAJOR)
if(ROCPROFILER_SDK_DXG_TOPOLOGY_SUPPORTED)
    foreach(_TARGET hsakmt::hsakmt rocprofiler-sdk-hsakmt-nolink)
        if(TARGET ${_TARGET})
            foreach(_PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                              INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
                get_target_property(_DIRS ${_TARGET} ${_PROPERTY})
                foreach(_DIR ${_DIRS})
                    if(NOT _ROCDXG_VERSION_MAJOR AND EXISTS
                                                     "${_DIR}/hsakmt/rocdxg_version.h")
                        file(STRINGS "${_DIR}/hsakmt/rocdxg_version.h"
                             _ROCDXG_VERSION_MAJOR
                             REGEX "^#define ROCDXG_LIB_VERSION_MAJOR [0-9]+")
                        string(
                            REGEX
                            REPLACE "^#define ROCDXG_LIB_VERSION_MAJOR ([0-9]+).*" "\\1"
                                    _ROCDXG_VERSION_MAJOR "${_ROCDXG_VERSION_MAJOR}")
                    endif()
                endforeach()
            endforeach()
        endif()
    endforeach()
endif()

if(ROCPROFILER_SDK_DXG_TOPOLOGY_SUPPORTED)
    list(APPEND ROCPROFILER_LIB_PLATFORM_TEST_SOURCES wsl_dxg_topology.cpp
         wsl_dxg_thunk.cpp)
endif()

if(_ROCDXG_VERSION_MAJOR)
    list(APPEND ROCPROFILER_LIB_PLATFORM_TEST_SOURCES wsl_dxg_loader.cpp)

    # wsl_dxg_loader.cpp finds these beside platform-dispatch-tests, in the build tree and
    # the install tree alike.
    set(_WSL_DXG_LOADER_TARGETS
        wsl-dxg-loader-probe wsl-dxg-fixture-thunk wsl-dxg-fixture-runpath-caller
        wsl-dxg-fixture-rpath-caller wsl-dxg-fixture-plain-caller)

    # Not linked against the HSA runtime: a real libhsa-runtime64 mapped into the probe
    # would be one more directory for the lookup under test to consider.
    add_executable(wsl-dxg-loader-probe wsl_dxg_loader_probe.cpp)
    target_link_libraries(
        wsl-dxg-loader-probe PRIVATE rocprofiler-sdk::rocprofiler-sdk-static-library
                                     rocprofiler-sdk::rocprofiler-sdk-common-library)

    # Like librocdxg itself: the versioned soname and -z nodelete
    # (libhsakmt/CMakeLists_wsl.txt).
    add_library(wsl-dxg-fixture-thunk SHARED wsl_dxg_loader_thunk_fixture.cpp)
    target_link_libraries(wsl-dxg-fixture-thunk
                          PRIVATE rocprofiler-sdk::rocprofiler-sdk-hsakmt-nolink)
    target_link_options(
        wsl-dxg-fixture-thunk PRIVATE
        "LINKER:-soname,librocdxg.so.${_ROCDXG_VERSION_MAJOR}" "LINKER:-z,nodelete")
    set_target_properties(wsl-dxg-fixture-thunk PROPERTIES NO_SONAME ON INSTALL_RPATH "")

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
    set_target_properties(wsl-dxg-fixture-plain-caller PROPERTIES INSTALL_RPATH "")

    set_target_properties(
        wsl-dxg-fixture-thunk wsl-dxg-fixture-runpath-caller wsl-dxg-fixture-rpath-caller
        wsl-dxg-fixture-plain-caller
        PROPERTIES BUILD_WITH_INSTALL_RPATH ON
                   INSTALL_RPATH_USE_LINK_PATH OFF
                   LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    install(
        TARGETS ${_WSL_DXG_LOADER_TARGETS}
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}/tests/unit-tests/bin
        COMPONENT tests)
elseif(ROCPROFILER_SDK_DXG_TOPOLOGY_SUPPORTED)
    message(
        STATUS "hsakmt/rocdxg_version.h not found: not building the wsl_dxg_loader tests")
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
if(_WSL_DXG_LOADER_TARGETS)
    add_dependencies(platform-dispatch-tests ${_WSL_DXG_LOADER_TARGETS})
endif()

# Reuses the sysfs topology fixture from lib/tests so the gnulinux force-platform test can
# resolve a real node tree on bare-metal CI. The sysfs topology should already exist in
# the build directory so do not copy over the data again, especially since
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
