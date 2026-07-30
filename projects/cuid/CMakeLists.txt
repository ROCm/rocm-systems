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

#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 3.14)

set(AMDCUID "amdcuid" CACHE INTERNAL "")
set(AMDCUID_PACKAGE ${AMDCUID} CACHE STRING "")

# ROCM_DIR should be passed in via command line
set(ROCM_DIR "/opt/rocm" CACHE PATH "ROCm directory.")

# Default libdir to "lib", this skips GNUInstallDirs from trying to take a guess if it's unset
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")

# Read version from amd_cuid.h so there is a single source of truth
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/lib/include/amd_cuid.h" _CUID_HEADER)
string(
    REGEX MATCH "#define AMDCUID_LIB_VERSION_MAJOR ([0-9]+)"
    _
    "${_CUID_HEADER}"
)
set(VERSION_MAJOR "${CMAKE_MATCH_1}")
string(
    REGEX MATCH "#define AMDCUID_LIB_VERSION_MINOR ([0-9]+)"
    _
    "${_CUID_HEADER}"
)
set(VERSION_MINOR "${CMAKE_MATCH_1}")
string(
    REGEX MATCH "#define AMDCUID_LIB_VERSION_PATCH ([0-9]+)"
    _
    "${_CUID_HEADER}"
)
set(VERSION_PATCH "${CMAKE_MATCH_1}")
set(VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
message("Package version: ${VERSION_STRING}")

# Don't print 'Up-to-date' when installing
set(CMAKE_INSTALL_MESSAGE LAZY)

# This must go after some CMAKE_* variables
project(${AMDCUID} VERSION "${VERSION_STRING}")

# Usually ROCM tools are installed into /opt/rocm
# CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT is only available after project()
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX
            "C:/Program Files/AMD/ROCM/AMDCUID"
            CACHE PATH
            "Default installation directory."
            FORCE
        )
    else() # Linux/Unix
        set(CMAKE_INSTALL_PREFIX
            "${ROCM_DIR}/core"
            CACHE PATH
            "Default installation directory."
            FORCE
        )
    endif()
endif()

# Include CMAKE_INSTALL_* variables
# This must go after project()
include(GNUInstallDirs)

# C++ standard for all targets
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# When cmake -DBUILD_TESTS off by default, it will not build the CUID test suite
option(BUILD_TESTS "Build test suite" OFF)
# When cmake -DBUILD_EXAMPLES off by default, it will not build the CUID examples
option(BUILD_EXAMPLES "Build examples" OFF)
# When cmake -DBUILD_DAEMON off by default, it will not build the CUID daemon
option(BUILD_DAEMON "Build daemon" OFF)

# Version-independent directory shared by all installed versions.
# The HMAC key and daemon config file are stored here so that multi-version
# ROCm installs all agree on the same key and settings, keeping CUIDs stable
# across version upgrades.  The path must NOT contain CMAKE_INSTALL_PREFIX
# (which is version-specific, e.g. /opt/rocm-6.1/core).
if(WIN32)
    set(AMDCUID_SHARED_CONFIG_DIR
        "C:/ProgramData/AMD/AMDCUID/config"
        CACHE PATH
        "Version-independent directory for the HMAC key and daemon config."
    )
else()
    set(AMDCUID_SHARED_CONFIG_DIR
        "/etc/amdcuid"
        CACHE PATH
        "Version-independent directory for the HMAC key and daemon config."
    )
endif()

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
if(BUILD_DAEMON)
    add_subdirectory(daemon)
endif()
if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif()
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Unified post-install wrapper: runs the library post-install always, and the
# daemon post-install automatically when the daemon binary is present.
if(UNIX AND NOT APPLE)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/amdcuid_postinst_wrapper.sh.in
        ${CMAKE_CURRENT_BINARY_DIR}/amdcuid_postinst.sh
        @ONLY
    )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/amdcuid_prerm_wrapper.sh.in
        ${CMAKE_CURRENT_BINARY_DIR}/amdcuid_prerm.sh
        @ONLY
    )
    install(
        PROGRAMS
            ${CMAKE_CURRENT_BINARY_DIR}/amdcuid_postinst.sh
            ${CMAKE_CURRENT_BINARY_DIR}/amdcuid_prerm.sh
        DESTINATION ${CMAKE_INSTALL_DATADIR}/${AMDCUID}
        COMPONENT ${LIB_COMPONENT}
    )
endif()

# Uninstall target
# NOTE: /etc/amdcuid/ (HMAC key + config) is intentionally left in place by
#       the prerm script to preserve existing CUID values across reinstalls.
#       Remove manually with: sudo rm -rf /etc/amdcuid
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE
        @ONLY
    )
    add_custom_target(
        uninstall
        COMMAND
            ${CMAKE_COMMAND} -P
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        COMMENT "Uninstalling ${AMDCUID} ..."
    )
endif()
