# Required Defines first:
option(INSTALL_GTEST "Install GTest (only useful if GTest is not already installed)" OFF)

# Help tests find libraries at runtime
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--enable-new-dtags")
set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib"
    CACHE STRING "RUNPATH for tests. Helps find libgtest.so and libamd_smi.so")

# Download and compile googletest
include(FetchContent)
FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.14.0)
FetchContent_MakeAvailable(googletest)

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

# Other source directories
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/functional functionalSources)

set(TEST "amdsmitst")

# Source files
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} tstSources)

# Header file include path
include_directories(${TEST} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${ROCM_INC_DIR}/..)

# Build rules
add_executable(${TEST} ${tstSources} ${functionalSources})
target_link_libraries(${TEST} ${AMD_SMI} GTest::gtest_main c stdc++ pthread)

# Install tests
install(
    TARGETS ${TEST}
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests
    COMPONENT ${TESTS_COMPONENT})

install(
    FILES amdsmitst.exclude
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests
    COMPONENT ${TESTS_COMPONENT})

# Install googletest libraries with tests
install(
    TARGETS gtest gtest_main
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests
    COMPONENT ${TESTS_COMPONENT})
