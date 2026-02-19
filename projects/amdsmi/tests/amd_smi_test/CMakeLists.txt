option(INSTALL_GTEST "Install GTest" OFF)

# Help tests find libraries at runtime
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--enable-new-dtags")
set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib"
    CACHE STRING "RUNPATH for tests. Helps find libgtest.so and libamd_smi.so")

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

# Other source directories
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/functional functionalSources)

set(TEST "amdsmitst")

# Source files
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} tstSources)

# Header file include path
include_directories(${TEST} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${ROCM_INC_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../../)

set(AMDSMINIC_LIB "amdsminic")
set(AMDSMINIC_PATH "${CMAKE_BINARY_DIR}/src/ai-nic/lib${AMDSMINIC_LIB}.a")

# Build rules
add_executable(${TEST} ${tstSources} ${functionalSources})
target_link_libraries(${TEST} ${AMD_SMI} GTest::gtest c stdc++ pthread ${AMDSMINIC_PATH} ${FILESYSTEM_LIB})

# Install tests
install(
    TARGETS ${TEST}
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests
    COMPONENT ${TESTS_COMPONENT})

install(
    FILES amdsmitst.exclude
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests
    COMPONENT ${TESTS_COMPONENT})
