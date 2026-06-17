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

cmake_minimum_required(VERSION 3.14)
project(amdcuid)

# Set platform-appropriate default install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX
            "C:/Program Files/AMD/AMDCUID"
            CACHE PATH
            "Installation prefix"
            FORCE
        )
    else() # Linux/Unix
        set(CMAKE_INSTALL_PREFIX
            "/opt/amdcuid"
            CACHE PATH
            "Installation prefix"
            FORCE
        )
    endif()
endif()

include(GNUInstallDirs)

if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # DLLs
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # Import libs
endif()

# Platform-specific dependencies; use native Windows bcrypt for HMAC, and OpenSSL on Linux/Unix
if(WIN32)
    # Use the native Windows bcrypt library and link it directly on Windows targets.
else()
    # Prefer a system-installed OpenSSL; fall back to building from source via
    # FetchContent if the dev package is not present (e.g. in bare containers).
    find_package(OpenSSL QUIET)
    if(NOT OPENSSL_FOUND)
        message(
            STATUS
            "OpenSSL not found on system; fetching and building from source"
        )
        include(FetchContent)
        FetchContent_Declare(
            openssl-cmake
            GIT_REPOSITORY https://github.com/janbar/openssl-cmake.git
            GIT_TAG 1.1.1w-20231130
        )
        FetchContent_MakeAvailable(openssl-cmake)
    endif()
endif()

message("Build Configuration:")
message("-----------ROCM_DIR : " ${ROCM_DIR})
message("------Install Prefix: " ${CMAKE_INSTALL_PREFIX})
message("--------Build Type  : " ${CMAKE_BUILD_TYPE})

# Define all the install component labels
set(LIB_COMPONENT "library")
set(DAEMON_COMPONENT "daemon")
set(CLI_COMPONENT "cli")
set(EXAMPLE_COMPONENT "example")
set(TESTS_COMPONENT "tests")

add_subdirectory(lib)
add_subdirectory(cli)
add_subdirectory(daemon)
add_subdirectory(example)

option(BUILD_TESTS "Build test suite" OFF)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ==============================================================================
# CPack Configuration for Portable Packages
# ==============================================================================
set(CPACK_PACKAGE_NAME "amdcuid")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AMD Compute Unit ID Library and Tools")

# Use the project's install prefix for all package types
set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
)
set(CPACK_PACKAGE_CONTACT "support@amd.com")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

# Component-based installation
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "Runtime")
set(CPACK_COMPONENT_LIBRARY_DISPLAY_NAME "Library")
set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "Development")

# Platform-specific generators
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
else()
    set(CPACK_GENERATOR "TGZ;DEB;RPM")

    # DEB-specific
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_CONTACT}")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libssl1.1 | libssl3")
    set(CPACK_DEBIAN_PACKAGE_SECTION "utils")

    # RPM-specific
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_PACKAGE_GROUP "System Environment/Libraries")
    set(CPACK_RPM_PACKAGE_REQUIRES "openssl-libs")
endif()

include(CPack)
