if(NOT DEFINED PROJECT_NAME)
    cmake_minimum_required(VERSION 3.15)
    project(rdc_tests LANGUAGES C CXX)
endif()

# When building out-of-tree we need the project root to locate headers
if(NOT DEFINED RDC_SOURCE_DIR)
    # tests/rdc_tests -> .. -> tests -> .. -> <rdc root>
    get_filename_component(RDC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
endif()

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake RDC test                              ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

if(WIN32)
    message("rdc library test suite is not supported on Windows platform")
    return()
endif()

if(NOT DEFINED ROCM_DIR)
    set(ROCM_DIR "/opt/rocm" CACHE PATH "ROCm install directory.")
endif()
if(NOT DEFINED AMD_SMI_INCLUDE_DIR)
    set(AMD_SMI_INCLUDE_DIR "${ROCM_DIR}/include" CACHE PATH "Path to AMD SMI headers.")
endif()
if(NOT DEFINED AMD_SMI_LIB_DIR)
    set(AMD_SMI_LIB_DIR "${ROCM_DIR}/lib" CACHE PATH "Path to AMD SMI libs.")
endif()
if(NOT DEFINED ROCM_INSTALL_DIR)
    set(ROCM_INSTALL_DIR "${ROCM_DIR}" CACHE PATH "Root used for ROCm runtime libraries.")
endif()

# Required Defines first:
option(INSTALL_GTEST "Install GTest (only useful if GTest is not already installed)" OFF)

# Hack to find libraries after installation
# /opt/rocm/share/rdc/rdctst_tests/../../../ = /opt/rocm
set(RDCTST_RPATH "\$ORIGIN" "\$ORIGIN/../../../lib" "\$ORIGIN/../../../lib/rdc")
# combine lists
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${RDCTST_RPATH})

#
# Print out the build configuration being used:
#
message("")
message("Build Configuration:")
message("-----------BuildType: " ${BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("------------ROCM_DIR: " ${ROCM_DIR})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("--------SMI Lib Dir: " ${AMD_SMI_LIB_DIR})
message("--------SMI Inc Dir: " ${AMD_SMI_INCLUDE_DIR})
message("")

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(RDCTST "rdctst")

# Prefer system GTest when available (useful in offline envs); otherwise FetchContent.
find_package(GTest QUIET)
if(GTest_FOUND)
    message(STATUS "Using system GTest")
    set(GTEST_TARGET GTest::gtest_main)
    set(GTEST_INSTALL_TARGETS)  # system package, do not install
else()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    set(GTEST_TARGET GTest::gtest_main)
    set(GTEST_INSTALL_TARGETS gtest gtest_main)
endif()

# Source files
set(rdctstSources
    ${SRC_DIR}/main.cc
    ${SRC_DIR}/test_base.cc
    ${SRC_DIR}/test_common.cc
    ${SRC_DIR}/test_utils.cc
)

# Functional test source files
set(functionalSources
    ${SRC_DIR}/functional/rdci_discovery.cc
    ${SRC_DIR}/functional/rdci_dmon.cc
    ${SRC_DIR}/functional/rdci_fieldgroup.cc
    ${SRC_DIR}/functional/rdci_group.cc
    ${SRC_DIR}/functional/rdci_stats.cc
    ${SRC_DIR}/functional/rdci_cpu_support.cc
)

# When built out of tree, consume installed RDC libs instead of rebuilding them.
if(NOT TARGET rdc_bootstrap OR NOT TARGET rdc)
    find_package(rdc CONFIG REQUIRED)
endif()
set(RDC_TEST_DEPS rdc_bootstrap rdc)

link_directories(${ROCM_INSTALL_DIR} ${AMD_SMI_LIB_DIR})

# Build rules
add_executable(${RDCTST} ${rdctstSources} ${functionalSources})

# Header file include path
target_include_directories(
    ${RDCTST}
    PUBLIC ${RDC_SOURCE_DIR}/include
    PUBLIC ${AMD_SMI_INCLUDE_DIR}
    PUBLIC ${SRC_DIR}/..
)

target_link_libraries(
    ${RDCTST}
    PUBLIC ${RDC_TEST_DEPS}
    PUBLIC ${GTEST_TARGET}
    PUBLIC c
    PUBLIC stdc++
    PUBLIC pthread
)

install(
    TARGETS ${RDCTST} ${GTEST_INSTALL_TARGETS}
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)

install(
    FILES ${RDCTST}.exclude
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)
