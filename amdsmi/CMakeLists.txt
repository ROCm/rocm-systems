# SPDX-License-Identifier: MIT
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
cmake_minimum_required(VERSION 3.20)

set(AMD_SMI "amd_smi")
set(AMD_SMI_LIBS_TARGET "${AMD_SMI}_lib")
set(CPACK_PACKAGE_NAME amd-smi-lib CACHE STRING "")

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared library (.so) or not.")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" CACHE INTERNAL "Default module path.")
## Include common cmake modules
include(utils)
include(help_package)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules/package_version.cmake)
find_package(PkgConfig)

generic_add_rocm()

# provide git to utilities
find_program(GIT NAMES git)

# Set Lintian Support Flags
set(BUILD_ENABLE_LINTIAN_OVERRIDES
    OFF
    CACHE BOOL
    "Enable/Disable Lintian Overrides"
)
set(BUILD_DEBIAN_PKGING_FLAG
    OFF
    CACHE BOOL
    "Internal Status Flag to indicate Debian Packaging Build"
)

# Single source of truth for the package version. Callers may override by
# passing -DPACKAGE_VERSION=x.y.z; otherwise the value is read from the
# shared VERSION file at the repository root.
if(NOT DEFINED PACKAGE_VERSION OR "${PACKAGE_VERSION}" STREQUAL "")
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" PACKAGE_VERSION
         LIMIT_COUNT 1)
    string(STRIP "${PACKAGE_VERSION}" PACKAGE_VERSION)
endif()
set(AMDSMI_PACKAGE_VERSION "${PACKAGE_VERSION}")

# Parse AMDSMI_PACKAGE_VERSION as the single source of truth
get_package_version_number(${AMDSMI_PACKAGE_VERSION})
set(AMDSMI_PACKAGE_VERSION_BASE "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
message("Package version: ${AMDSMI_PACKAGE_VERSION}")
set(AMDSMI_VERSION_MAJOR "${CPACK_PACKAGE_VERSION_MAJOR}")
set(${AMD_SMI_LIBS_TARGET}_VERSION_MAJOR "${CPACK_PACKAGE_VERSION_MAJOR}")
set(${AMD_SMI_LIBS_TARGET}_VERSION_MINOR "${CPACK_PACKAGE_VERSION_MINOR}")
set(${AMD_SMI_LIBS_TARGET}_VERSION_PATCH "${CPACK_PACKAGE_VERSION_PATCH}")
# Used by configure_file() in py-interface and amdsmi_cli for _version.py, setup.py, pyproject.toml
set(${AMD_SMI_LIBS_TARGET}_VERSION_STRING "${AMDSMI_PACKAGE_VERSION_BASE}")

# The following default version values should be updated as appropriate for
# ABI breaks (update MAJOR and MINOR), and ABI/API additions (update MINOR).
# Until ABI stabilizes VERSION_MAJOR will be 0. This should be over-ridden
# by git tags (through "git describe") when they are present.
set(PKG_VERSION_MAJOR "${CPACK_PACKAGE_VERSION_MAJOR}")
set(PKG_VERSION_MINOR "${CPACK_PACKAGE_VERSION_MINOR}")
set(PKG_VERSION_PATCH "${CPACK_PACKAGE_VERSION_PATCH}")

project(${AMD_SMI_LIBS_TARGET} DESCRIPTION "AMD System Management libraries")

# Set default installation prefix to /opt/rocm
if(NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Installation prefix" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Link with stdc++fs for filesystem support (only for GCC < 9.0)
set(FILESYSTEM_LIB "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        set(FILESYSTEM_LIB stdc++fs)
        message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION} detected, linking with stdc++fs for filesystem support")
    endif()
endif()

include(GNUInstallDirs)

# Force ENABLE_ESMI_LIB ON so that Python scripts can resolve ESMI-related
# function symbols.  Actual ESMI functionality is not available on this platform.
set(ENABLE_ESMI_LIB ON)
add_definitions("-DENABLE_ESMI_LIB=1")

option(BUILD_TESTS "Build test suite" OFF)
option(ENABLE_ASAN_PACKAGING "" OFF)

# If amdsmi is built as a static library, it should support being embedded in other programs. The setting below essentially enables the -fPIC flag.
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Enable position independent code for all targets")
mark_as_advanced(CMAKE_POSITION_INDEPENDENT_CODE)

include(CMakeDependentOption)
# these options don't work without BUILD_SHARED_LIBS
cmake_dependent_option(BUILD_WRAPPER "Rebuild AMDSMI-wrapper" OFF "BUILD_SHARED_LIBS" OFF)
cmake_dependent_option(BUILD_CLI "Build AMDSMI-CLI and install" ON "BUILD_SHARED_LIBS" OFF)
cmake_dependent_option(BUILD_RUST_WRAPPER "Build rust wrapper and install" OFF "BUILD_SHARED_LIBS" OFF)
cmake_dependent_option(ENABLE_LDCONFIG "Set library links and caches using ldconfig." ON "BUILD_SHARED_LIBS" OFF)

# Set share path here because project name != amd_smi
set(SHARE_INSTALL_PREFIX "${CMAKE_INSTALL_DATAROOTDIR}/${AMD_SMI}" CACHE STRING "Tests and Example install directory")

# Packaging directives
set(PKG_MAINTAINER_NM "AMD-SMILib Support")
set(PKG_MAINTAINER_EMAIL "amd-smi.support@amd.com")
set(CPACK_PACKAGE_CONTACT
    "${PKG_MAINTAINER_NM} <${PKG_MAINTAINER_EMAIL}>"
    CACHE STRING
    ""
)

generic_package()

# Dependencies
find_package(Threads REQUIRED)


configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/config/amd_smi_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/config/amd_smi_config.h"
    @ONLY
)

## Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fno-rtti")
if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -msse -msse2")
endif()
# Security options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion -Wcast-align")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=2 -fno-common -Wstrict-overflow")
# Intentionally leave out -Wsign-promo. It causes spurious warnings.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Woverloaded-virtual -Wreorder")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/shared_mutex
    ${CMAKE_CURRENT_SOURCE_DIR}/include/amd_smi
    ${ESMI_INC_DIR}
)

set(CMN_SRC_LIST)

set(CMN_INC_LIST)

if(NOT TARGET dxg_shared)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../shared
                     ${CMAKE_CURRENT_BINARY_DIR}/shared)
endif()

add_subdirectory("src")

if(BUILD_TESTS)
    set(TESTS_COMPONENT "tests")
    add_subdirectory("tests/amd_smi_test")
    add_subdirectory("tests/python_unittest")
endif()

# python interface, CLI, and py-test depend on shared libraries
if(BUILD_SHARED_LIBS)
    add_subdirectory("py-interface")
    if(BUILD_CLI)
        add_subdirectory("amdsmi_cli")
    endif()
endif()


include(CMakePackageConfigHelpers)

configure_package_config_file(
    amd_smi-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/amd_smi-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${AMD_SMI}
    PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_BINDIR)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/amd_smi-config-version.cmake
                                 VERSION "${AMDSMI_PACKAGE_VERSION_BASE}" COMPATIBILITY SameMajorVersion)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/amd_smi-config.cmake ${CMAKE_CURRENT_BINARY_DIR}/amd_smi-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${AMD_SMI}
    COMPONENT dev)

# Create cmake target (shared library only)
if(BUILD_SHARED_LIBS)
# Add all targets to the build-tree export set
export(TARGETS ${AMD_SMI} FILE "${PROJECT_BINARY_DIR}/amd_smi_target.cmake")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE ${AMD_SMI})

install(
    EXPORT amd_smiTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${AMD_SMI}
    COMPONENT dev)
endif()

if(ENABLE_ASAN_PACKAGING)
    set(COMP_TYPE "asan")
    # install license file in share/doc/amd_smi-asan folder
    install(
        FILES ${CPACK_RESOURCE_FILE_LICENSE}
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${CPACK_PACKAGE_NAME}-asan
        RENAME LICENSE.txt
        COMPONENT asan)
else()
    set(COMP_TYPE "dev")
endif()

# docs are installed into different share directory from tests and examples
install(
    FILES ${CPACK_RESOURCE_FILE_LICENSE}
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${CPACK_PACKAGE_NAME}
    RENAME LICENSE.txt
    COMPONENT dev)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${CPACK_PACKAGE_NAME}
    COMPONENT dev)


#Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "python3-argcomplete")
set(CPACK_DEBIAN_ASAN_PACKAGE_RECOMMENDS ${CPACK_DEBIAN_PACKAGE_RECOMMENDS})
set(CPACK_DEBIAN_DEV_PACKAGE_RECOMMENDS ${CPACK_DEBIAN_PACKAGE_RECOMMENDS})
set(CPACK_DEBIAN_PACKAGE_DEPENDS "sudo, libc6, python3 (>= 3.6.8), python3-pip, python3-setuptools, python3-wheel")
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})

# $CURRENT_YEAR is used by copyright.in
string(TIMESTAMP CURRENT_YEAR "%Y")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/copyright.in DEBIAN/copyright @ONLY)

## Process the Debian install/remove scripts to update the CPACK variables
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst.in
    DEBIAN/postinst
    @ONLY
    FILE_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm.in
    DEBIAN/prerm
    @ONLY
    FILE_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)
list(APPEND CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_BINARY_DIR}/DEBIAN/postinst"
     "${CMAKE_CURRENT_BINARY_DIR}/DEBIAN/prerm")

# Configure pre-rm for tests only
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/${CPACK_PACKAGE_NAME}-tests/prerm.in
    DEBIAN/${CPACK_PACKAGE_NAME}-tests/prerm
    @ONLY
    FILE_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)

# Assign control scripts to the AMDSMI Lib & Tests packages
set(CPACK_DEBIAN_DEV_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_BINARY_DIR}/DEBIAN/prerm")
set(CPACK_DEBIAN_TESTS_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_BINARY_DIR}/DEBIAN/${CPACK_PACKAGE_NAME}-tests/prerm")

# install copyright file into share/doc/amd-smi-lib/copyright
# required for debian package compliance
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/DEBIAN/copyright"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${CPACK_PACKAGE_NAME}
    COMPONENT dev)

# Add rocm-core dependency if -DROCM_DEP_ROCMCORE=ON is passed
if(ROCM_DEP_ROCMCORE)
    string(APPEND CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ", rocm-core-asan")
    string(APPEND CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ", rocm-core")
    string(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS ", rocm-core")
endif()

## Enable Component Mode and set component specific flags
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
set(CPACK_DEBIAN_TESTS_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-tests")
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-asan")
if(ENABLE_ASAN_PACKAGING)
    # ASAN Package requires only asan component with libraries and license file
    set(CPACK_COMPONENTS_ALL asan)
else()
    set(CPACK_COMPONENTS_ALL dev tests)
endif()

#Set the names now using CPACK utility
set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

apply_package_version_suffix("${AMDSMI_PACKAGE_VERSION_BASE}" CPACK_PACKAGE_VERSION)

# Configure Lintian Specific Package Data
configure_pkg( ${CPACK_PACKAGE_NAME} ${COMP_TYPE} ${CPACK_PACKAGE_VERSION} ${PKG_MAINTAINER_NM} ${PKG_MAINTAINER_EMAIL})

# Custom installation for Debian Lintian File
if(BUILD_ENABLE_LINTIAN_OVERRIDES AND BUILD_DEBIAN_PKGING_FLAG)
    set(OVERRIDE_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/DEBIAN/${DEB_OVERRIDES_INSTALL_FILENM}"
    )
    set(OVERRIDE_TEMP_INSTALL_LOC
        "/_CPack_Packages/Linux/DEB/${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-Linux/runtime"
    )
    if(BUILD_SHARED_LIBS)
        set(CPACK_INSTALL_COMMANDS
            "${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}${OVERRIDE_TEMP_INSTALL_LOC}${DEB_OVERRIDES_INSTALL_PATH}"
            "${CMAKE_COMMAND} -E copy ${OVERRIDE_FILE} ${CMAKE_BINARY_DIR}${OVERRIDE_TEMP_INSTALL_LOC}${DEB_OVERRIDES_INSTALL_PATH}"
        )
    else()
        set(OVERRIDE_FILE_STATIC "${DEB_OVERRIDES_INSTALL_FILENM}-static-dev")
        set(OVERRIDE_FILE_STATIC_DEST
            "${CMAKE_BINARY_DIR}${OVERRIDE_TEMP_INSTALL_LOC}${DEB_OVERRIDES_INSTALL_PATH}${OVERRIDE_FILE_STATIC}"
        )
        set(CPACK_INSTALL_COMMANDS
            "${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}${OVERRIDE_TEMP_INSTALL_LOC}${DEB_OVERRIDES_INSTALL_PATH}"
            "${CMAKE_COMMAND} -E copy ${OVERRIDE_FILE} ${OVERRIDE_FILE_STATIC_DEST}"
        )
    endif()
endif()

# If rocdxg is also being built, hook its CPack into the built-in 'package' target
# by appending a post-build script. This makes 'make package' generate both packages.
if(BUILD_ROCDXG)
    list(APPEND CPACK_POST_BUILD_SCRIPTS "${CMAKE_BINARY_DIR}/rocdxg_cpack_post.cmake")
endif()

include(CPack)

generic_package_post()
