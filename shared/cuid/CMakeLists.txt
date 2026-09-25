# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#
# Minimum version of cmake required
#
# 3.21 for $<TARGET_OBJECTS:> on a non-OBJECT library, which lib/CMakeLists.txt
# uses to fold the shared SHA-256 into the installed archive. The rest of
# rocm-systems is on 3.20 (projects/amdsmi) and 3.14 was older than anything the
# repository is actually built with.
cmake_minimum_required(VERSION 3.21)

set(AMDCUID "amdcuid" CACHE INTERNAL "")
set(AMDCUID_PACKAGE ${AMDCUID} CACHE STRING "")

# ROCM_DIR should be passed in via command line
set(ROCM_DIR "/opt/rocm" CACHE PATH "ROCm directory.")

# ROCm installs under lib, not lib64. Top level only: the cache is global to a
# build tree, so seeding CMAKE_INSTALL_LIBDIR from a subdirectory would decide
# the layout of every sibling project in a superbuild.
if(NOT DEFINED CMAKE_INSTALL_LIBDIR AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CMAKE_INSTALL_LIBDIR "lib" CACHE PATH "Library install directory")
endif()

# Read version from amd_cuid.h so there is a single source of truth
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/lib/include/amd_cuid.h" _CUID_HEADER)
string(REGEX MATCH "#define AMDCUID_LIB_VERSION_MAJOR ([0-9]+)" _ "${_CUID_HEADER}")
set(VERSION_MAJOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define AMDCUID_LIB_VERSION_MINOR ([0-9]+)" _ "${_CUID_HEADER}")
set(VERSION_MINOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define AMDCUID_LIB_VERSION_PATCH ([0-9]+)" _ "${_CUID_HEADER}")
set(VERSION_PATCH "${CMAKE_MATCH_1}")
set(VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
message("Package version: ${VERSION_STRING}")

# Don't print 'Up-to-date' when installing
set(CMAKE_INSTALL_MESSAGE LAZY)

# This must go after some CMAKE_* variables
project(${AMDCUID} VERSION "${VERSION_STRING}")

# Usually ROCM tools are installed into /opt/rocm.
# CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT is only available after project().
# Top level only: forcing the prefix from a subdirectory relocates the parent's install.
if(PROJECT_IS_TOP_LEVEL AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "C:/Program Files/AMD/ROCM/AMDCUID" CACHE PATH "Default installation directory." FORCE)
    else() # Linux/Unix
        set(CMAKE_INSTALL_PREFIX "${ROCM_DIR}/core" CACHE PATH "Default installation directory." FORCE)
    endif()
endif()

# Include CMAKE_INSTALL_* variables
# This must go after project()
include(GNUInstallDirs)

# C++ standard for all targets
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# A parent project has its own BUILD_TESTS etc. (projects/amdsmi sets
# BUILD_TESTS=ON), which option() would silently adopt. As a subdirectory, build
# only the library unless the parent sets the AMDCUID_-prefixed names.
if(PROJECT_IS_TOP_LEVEL)
    # When cmake -DBUILD_TESTS off by default, it will not build the CUID test suite
    option(BUILD_TESTS "Build test suite" OFF)
    # When cmake -DBUILD_EXAMPLES off by default, it will not build the CUID examples
    option(BUILD_EXAMPLES "Build examples" OFF)
else()
    set(BUILD_TESTS ${AMDCUID_BUILD_TESTS})
    set(BUILD_EXAMPLES ${AMDCUID_BUILD_EXAMPLES})
endif()

if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # DLLs
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # Import libs
endif()

# No third-party dependencies. SHA-256 and HMAC-SHA-256 are built from source
# out of shared/sha256; random key material comes from the platform CSPRNG
# (bcrypt on Windows, getrandom(2)//dev/urandom elsewhere).

message("Build Configuration:")
message("-----------ROCM_DIR : " ${ROCM_DIR})
message("------Install Prefix: " ${CMAKE_INSTALL_PREFIX})
message("--------Build Type  : " ${CMAKE_BUILD_TYPE})

# A parent that absorbs AMDCUID_IN_TREE_OBJECTS must not also ship a
# libamdcuid.a, headers or CMake package for a library nobody links.
option(AMDCUID_INSTALL "Install the library, headers and CMake package" ${PROJECT_IS_TOP_LEVEL})

# Define all the install component labels
set(LIB_COMPONENT "library")
set(EXAMPLE_COMPONENT "example")
set(TESTS_COMPONENT "tests")

add_subdirectory(lib)
if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif()
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# efivarfs shows AmdCuidKey world-readable and immutable. systemd-tmpfiles
# only reads fixed directories, never ones under an install prefix.
if(AMDCUID_INSTALL AND UNIX AND NOT APPLE)
    set(AMDCUID_TMPFILES_DIR "/usr/lib/tmpfiles.d" CACHE PATH "Directory systemd-tmpfiles reads amdcuid.conf from")
    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/tmpfiles.d/amdcuid.conf
        DESTINATION ${AMDCUID_TMPFILES_DIR}
        COMPONENT ${LIB_COMPONENT}
    )
endif()

# Uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE
        @ONLY
    )
    add_custom_target(
        uninstall
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        COMMENT "Uninstalling ${AMDCUID} ..."
    )
endif()
