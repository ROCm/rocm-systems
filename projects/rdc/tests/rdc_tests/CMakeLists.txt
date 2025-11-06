message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake RDC test                              ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

if(WIN32)
    message("rdc library test suite is not supported on Windows platform")
    return()
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
#   Build Src directory
#   Build Binary directory
#   Build Type: Debug Vs Release, 32 Vs 64
#   Compiler Version, etc
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

link_directories(${ROCM_INSTALL_DIR} ${AMD_SMI_LIB_DIR})

# Build rules
add_executable(${RDCTST} ${rdctstSources} ${functionalSources})

# Header file include path
target_include_directories(
    ${RDCTST}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${AMD_SMI_INCLUDE_DIR}
    PUBLIC ${SRC_DIR}/..
)

target_link_libraries(
    ${RDCTST}
    PUBLIC rdc_bootstrap
    PUBLIC rdc
    PUBLIC GTest::gtest
    PUBLIC c
    PUBLIC stdc++
    PUBLIC pthread
)

install(
    TARGETS ${RDCTST} gtest
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)

install(
    FILES ${RDCTST}.exclude
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)
