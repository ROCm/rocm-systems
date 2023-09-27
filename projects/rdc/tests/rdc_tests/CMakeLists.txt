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
set(RDCTST_RPATH
    "\$ORIGIN/../../../lib"
    "\$ORIGIN/../../../lib/rdc")
# replace lib with lib64
list(TRANSFORM RDCTST_RPATH REPLACE "lib" "lib64" OUTPUT_VARIABLE RDCTST_RPATH64)
# combine lists
set(CMAKE_INSTALL_RPATH
    ${CMAKE_INSTALL_RPATH}
    ${RDCTST_RPATH}
    ${RDCTST_RPATH64})

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
message("--------RSMI Lib Dir: " ${RSMI_LIB_DIR})
message("--------RSMI Inc Dir: " ${RSMI_INC_DIR})
message("")

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(RDCTST "rdctst")

# Download and compile googletest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)
FetchContent_MakeAvailable(googletest)

# Source files
aux_source_directory(${SRC_DIR} rdctstSources)

# Other source directories
aux_source_directory(${SRC_DIR}/functional functionalSources)

link_directories(${ROCM_INSTALL_DIR} ${RSMI_LIB_DIR})

# Build rules
add_executable(${RDCTST} ${rdctstSources} ${functionalSources})

# Header file include path
target_include_directories(
    ${RDCTST}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${RSMI_INC_DIR}
    PUBLIC ${SRC_DIR}/..)

target_link_libraries(${RDCTST}
    PUBLIC rdc_bootstrap
    PUBLIC rdc
    PUBLIC rdc_ras
    PUBLIC GTest::gtest_main
    PUBLIC c
    PUBLIC stdc++
    PUBLIC pthread)

install(TARGETS ${RDCTST} gtest gtest_main
    DESTINATION ${RDC_SHARE_INSTALL_PREFIX}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT})

install(FILES ${RDCTST}.exclude
    DESTINATION ${RDC_SHARE_INSTALL_PREFIX}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT})
