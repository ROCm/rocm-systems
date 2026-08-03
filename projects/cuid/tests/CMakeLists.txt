# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                     Cmake amdcuid test                            ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("------Install Prefix: " ${CMAKE_INSTALL_PREFIX})
message("-CMake inst. Bindir : " ${CMAKE_INSTALL_BINDIR})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("")

# Find required packages
find_package(GTest CONFIG QUIET)
# fallback to CMake module if GTest is not found via config
if(NOT GTest_FOUND)
    find_package(GTest MODULE QUIET)
endif()
# If GTest is still not found, fetch it from source.
# FetchContent defines GTest::gtest / GTest::gtest_main; find_package CONFIG
# mode does the same. find_package MODULE mode defines GTest::GTest / GTest::Main,
# so create aliases to normalize all three paths to a single target name.
if(NOT GTest_FOUND)
    message("GTest not found on system. Fetching from source...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.16.0
    )
    FetchContent_MakeAvailable(googletest)
elseif(TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    # MODULE mode found GTest::GTest — create the canonical aliases expected below.
    add_library(GTest::gtest ALIAS GTest::GTest)
endif()

set(AMDCUID_TST_EXECUTABLE amdcuid_test)

# Collect all test sources from the three source locations.
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} TEST_SOURCES_ROOT)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/unit TEST_SOURCES_UNIT)
aux_source_directory(
    ${CMAKE_CURRENT_SOURCE_DIR}/functional
    TEST_SOURCES_FUNCTIONAL
)

set(TEST_SOURCES
    ${TEST_SOURCES_ROOT}
    ${TEST_SOURCES_UNIT}
    ${TEST_SOURCES_FUNCTIONAL}
)

# Enable testing
enable_testing()

# Test executable
add_executable(${AMDCUID_TST_EXECUTABLE} ${TEST_SOURCES})

target_include_directories(
    ${AMDCUID_TST_EXECUTABLE}
    PRIVATE
        # Allows #include "unit/version_read_test.h" etc. from main.cc and test_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}
        # Allows #include "include/amd_cuid.h" and #include "src/cuid_util.h" etc.
        ${CMAKE_SOURCE_DIR}/lib
)

target_link_libraries(${AMDCUID_TST_EXECUTABLE} amdcuid_static GTest::gtest)

# Register two CTest targets split by privilege level.
add_test(
    NAME cuid_unprivileged_tests
    COMMAND ${AMDCUID_TST_EXECUTABLE} --gtest_filter=cuidtstUnprivileged.*
)
add_test(
    NAME cuid_privileged_tests
    COMMAND ${AMDCUID_TST_EXECUTABLE} --gtest_filter=cuidtstPrivileged.*
)

# Platform-specific config directory
if(WIN32)
    set(TESTS_INSTALL_DIR "TESTS") # Relative to prefix
else()
    set(TESTS_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/${AMDCUID}/tests")
endif()

# Install the test binary
install(
    TARGETS ${AMDCUID_TST_EXECUTABLE}
    DESTINATION ${TESTS_INSTALL_DIR}
    COMPONENT ${TESTS_COMPONENT}
)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake amdcuid_test                    ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
