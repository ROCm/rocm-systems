message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake RDC test                              ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

if(WIN32)
    message("rdc library test suite is not supported on Windows platform")
    return()
endif()

# Required Defines first:
option(INSTALL_GTEST "Install GTest (only useful if GTest is not already installed)" OFF)

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

# Try to find googletest
find_package(GTest 1.11.0)

# GTest_FOUND is set to TRUE if ANY version is found
# GTest_VERSION is set if 1.11.0 or newer version is found
if(NOT GTest_FOUND STREQUAL "TRUE" OR NOT DEFINED GTest_VERSION)
    # Google Test wasn't found. Download and compile ourselves
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0)
    FetchContent_MakeAvailable(googletest)
endif()

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

# TODO: figure out if this properly does runtime linking
target_link_libraries(${RDCTST}
    PUBLIC GTest::gtest_main
    PUBLIC rdc_bootstrap
    PUBLIC rdc
    PUBLIC c
    PUBLIC stdc++
    PUBLIC pthread)

install(TARGETS ${RDCTST}
    DESTINATION ${RDC_SHARE_INSTALL_PREFIX}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT})

install(FILES ${RDCTST}.exclude
    DESTINATION ${RDC_SHARE_INSTALL_PREFIX}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT})
