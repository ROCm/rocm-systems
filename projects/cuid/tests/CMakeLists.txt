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

# Find required packages
find_package(GTest CONFIG QUIET)
# fallback to CMake module if GTest is not found via config
if(NOT GTest_FOUND)
    find_package(GTest MODULE QUIET)
endif()
# If GTest is still not found, we'll fetch it from source in the next step.
if(NOT GTest_FOUND)
    # Download and compile googletest to ensure it's available
    message("GTest not found on system. Fetching from source...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

set(AMDCUID_TST_EXECUTABLE amdcuid_test)
set(UNIT_TEST_SOURCES
    unit/cuid_api_test.cc
    unit/cuid_misc_test.cc
    functional/cuid_reverse_lookup_test.cc
)

# Enable testing
enable_testing()

# Unit tests executable
add_executable(amdcuid_test ${UNIT_TEST_SOURCES})

target_include_directories(amdcuid_test PRIVATE ${CMAKE_SOURCE_DIR}/lib)

target_link_libraries(
    ${AMDCUID_TST_EXECUTABLE}
    amdcuid_static
    GTest::GTest
    GTest::Main
)

# Register tests with CTest
add_test(NAME cuid_unit_tests COMMAND ${AMDCUID_TST_EXECUTABLE})

# Platform-specific config directory
if(WIN32)
    set(TESTS_INSTALL_DIR "TESTS") # Relative to prefix
else()
    set(TESTS_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/amd_cuid/tests")
endif()

# Install the test binary
install(
    TARGETS ${AMDCUID_TST_EXECUTABLE}
    DESTINATION ${TESTS_INSTALL_DIR}
    COMPONENT tests
)
