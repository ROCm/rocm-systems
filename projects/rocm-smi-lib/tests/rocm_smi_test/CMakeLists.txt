message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    CMake ROCm SMI test                            ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

if(WIN32)
    message("rsmi library test suite is not supported on Windows platform")
    return()
endif()

# Required Defines first:
option(INSTALL_GTEST "Install GTest (only useful if GTest is not already installed)" OFF)

message("")
message("Build Configuration:")
message("-----------BuildType: " ${BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("")

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(RSMITST "rsmitst")

# Hack to find libraries after installation
# $ORIGIN is needed for libgtest.so
# /opt/rocm/share/rocm-smi/rsmitst_tests/../../../ = /opt/rocm
set(RSMITST_RPATH
    "\$ORIGIN/../../../lib"
    "\$ORIGIN")
# combine lists
set(CMAKE_INSTALL_RPATH
    ${CMAKE_INSTALL_RPATH}
    ${RSMITST_RPATH})

# TODO: Try to find googletest
# DISABLED because we want to install gtest with rocm_smi_lib ourselves
#find_package(GTest 1.12.0)

# GTest_FOUND is set to TRUE if ANY version is found
# GTest_VERSION is set if 1.12.0 or newer version is found
if(NOT GTest_FOUND STREQUAL "TRUE" OR NOT DEFINED GTest_VERSION)
    # Google Test wasn't found. Download and compile ourselves
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.0)
    FetchContent_MakeAvailable(googletest)
    install(TARGETS gtest gtest_main
        DESTINATION ${SHARE_INSTALL_PREFIX}/rsmitst_tests
        COMPONENT ${TESTS_COMPONENT})
endif()

# Other source directories
aux_source_directory(${SRC_DIR}/functional functionalSources)

# Source files for building rocrtst
aux_source_directory(${SRC_DIR} rsmitstSources)

# Build rules
add_executable(${RSMITST} ${rsmitstSources} ${functionalSources})

# Header file include path
target_include_directories(${RSMITST} PUBLIC ${SRC_DIR}/..)

target_link_libraries(
    ${RSMITST}
    PUBLIC ${ROCM_SMI_TARGET}
    PUBLIC gtest
    PUBLIC gtest_main
    PUBLIC c
    PUBLIC stdc++
    PUBLIC pthread
    PUBLIC dl)

install(TARGETS ${RSMITST}
    DESTINATION ${SHARE_INSTALL_PREFIX}/rsmitst_tests
    COMPONENT ${TESTS_COMPONENT})

install(FILES rsmitst.exclude
    DESTINATION ${SHARE_INSTALL_PREFIX}/rsmitst_tests
    COMPONENT ${TESTS_COMPONENT})
