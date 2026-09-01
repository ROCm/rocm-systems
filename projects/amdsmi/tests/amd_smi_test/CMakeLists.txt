# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

option(INSTALL_GTEST "Install GTest" OFF)

# Help tests find libraries at runtime
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--enable-new-dtags")
set(CMAKE_INSTALL_RPATH
    "\$ORIGIN:\$ORIGIN/../../../lib"
    CACHE STRING
    "RUNPATH for tests. Helps find libgtest.so and libamd_smi.so"
)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Use system GTest if available, otherwise fetch and build
find_package(GTest)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        # Originally mirrored from: https://github.com/google/googletest/releases/download/v1.16.0/googletest-1.16.0.tar.gz
        URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/googletest-1.16.0.tar.gz
        URL_HASH SHA256=78c676fc63881529bf97bf9d45948d905a66833fbfa5318ea2cd7478cb98f399
    )
    FetchContent_MakeAvailable(googletest)
endif()

enable_testing()

if(WIN32)
    message("amd_smi library test suite is not supported on Windows platform")
    return()
endif()

#
# Print out the build configuration being used:
#
#   Build Src directory
#   Build Binary directory
#   Build Type: Debug Vs Release
#   Compiler Version, etc
#
message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("")

# Collect all sources under unit/ and functional/ recursively. CONFIGURE_DEPENDS
# re-globs at build time, so a newly added test .cc is picked up without manually
# re-running cmake.
file(GLOB_RECURSE unitSources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cc)
file(GLOB_RECURSE functSources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/functional/*.cc)
if(NOT ENABLE_WSL_BACKEND)
    list(FILTER unitSources EXCLUDE REGEX "/unit/wsl/")
    list(FILTER functSources EXCLUDE REGEX "/functional/wsl/")
elseif(ENABLE_WSL_BACKEND)
    # functional/wsl/smi tests directly link rocdxg_smi_* symbols and require
    # librocdxg at build time. Drop them when librocdxg is not staged yet (e.g.
    # TheRock CI compiler-runtime stage, which builds before wsl-rocdxg).
    # unit/wsl tests only use the amdsmi API and are kept regardless.
    find_library(ROCDXG_LIB rocdxg)
    if(NOT ROCDXG_LIB)
        list(FILTER functSources EXCLUDE REGEX "/functional/wsl/")
    endif()
endif()

set(TEST "amdsmitst")

# Harness sources at the root of this directory. Listed explicitly (rather than
# aux_source_directory) so the set is stable and obvious; test sources live under
# unit/ and functional/ and are globbed above.
set(tstSources
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/test_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/test_common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/test_utils.cc
)

# Build rules
add_executable(${TEST} ${tstSources} ${unitSources} ${functSources})

# Header include paths, scoped to the test target (consistent with the
# target_* calls below). ${CMAKE_CURRENT_SOURCE_DIR} (= tests/amd_smi_test/) lets
# test_base.h, test_common.h, and test_utils.h be included as "test_base.h" from
# any subdirectory depth.
target_include_directories(
    ${TEST}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${ROCM_INC_DIR}/..
)

# Absolute path to the committed mock CPER fixtures (unit/gpu/mock_cper).
target_compile_definitions(${TEST} PRIVATE AMDSMI_TEST_MOCK_DIR="${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu/mock_cper")

if(ENABLE_WSL_BACKEND)
    target_compile_definitions(${TEST} PRIVATE ENABLE_WSL_BACKEND)
    target_include_directories(${TEST} PRIVATE ${PROJECT_SOURCE_DIR}/../rocr-runtime/libhsakmt/include)
    if(ROCDXG_LIB)
        target_link_libraries(${TEST} ${ROCDXG_LIB})
    endif()
endif()
target_link_libraries(
    ${TEST}
    ${AMD_SMI_TARGET}
    GTest::gtest
    c
    stdc++
    pthread
    ${AMDSMINIC_PATH}
    ${FILESYSTEM_LIB}
)

# Install tests
install(TARGETS ${TEST} DESTINATION ${SHARE_INSTALL_PREFIX}/tests COMPONENT ${TESTS_COMPONENT})

install(
    FILES amdsmitst.exclude detect_asic_filter.sh
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests
    COMPONENT ${TESTS_COMPONENT}
)

# CPER mock fixtures, installed next to the test binary so the packaged tests
# locate them at runtime instead of relying on the build-machine source path.
# Exclude __pycache__ so a locally-run sanitize_cper.py never leaks a stray .pyc
# into the package.
install(
    DIRECTORY unit/gpu/mock_cper
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests
    COMPONENT ${TESTS_COMPONENT}
    PATTERN "__pycache__" EXCLUDE
)

# Hardware-free unit test for the fwupd UMA carveout adapter. On a host without
# the fwupd carveout BIOS setting -- i.e. any CI machine: no APU firmware exposes
# it, and typically no fwupd daemon or D-Bus system bus is present -- both entry
# points must report NOT_SUPPORTED so the amdgpu sysfs node stays the sole UMA
# carveout interface. Kept in its own target/subdir so the aux_source_directory()
# sweep above does not pull it into ${TEST}.
add_executable(
    fwupd_carveout_test
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/fwupd_carveout_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/amd_smi/fwupd_carveout.cc
)
target_include_directories(
    fwupd_carveout_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${CMAKE_CURRENT_SOURCE_DIR}/../../src/amd_smi
)
target_link_libraries(fwupd_carveout_test GTest::gtest GTest::gtest_main ${CMAKE_DL_LIBS})
add_test(NAME fwupd_carveout_test COMMAND fwupd_carveout_test)
