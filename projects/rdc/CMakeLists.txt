# Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 3.15)

set(RDC "rdc" CACHE INTERNAL "")
set(RDC_PACKAGE ${RDC} CACHE STRING "")

# ROCM_DIR should be passed in via command line
set(ROCM_DIR "/opt/rocm" CACHE PATH "ROCm directory.")
if(DEFINED ROCM_PATH)
    message(
        WARNING
        "ROCM_PATH is not used by the build process! Did you mean to set ROCM_DIR instead?"
    )
endif()

# Default libdir to "lib", this skips GNUInstallDirs from trying to take a guess if it's unset:
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")

# Usually ROCM tools are installed into /opt/rocm
set(CMAKE_INSTALL_PREFIX ${ROCM_DIR} CACHE PATH "Default installation directory.")

set(CMAKE_BUILD_TYPE
    "RelWithDebInfo"
    CACHE STRING
    "Choose the type of build to perform: Debug, Release, RelWithDebInfo, MinSizeRel"
)
# Set the possible values of build type
set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel"
)

# Set compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -m64 -msse -msse2")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -DDEBUG"
    CACHE STRING
    "Flags for Debug builds"
)
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O2 -s -DNDEBUG"
    CACHE STRING
    "Flags for Release builds"
)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g -DNDEBUG"
    CACHE STRING
    "Flags for RelWithDebInfo builds"
)
set(CMAKE_CXX_FLAGS_MINSIZEREL
    "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os -DNDEBUG"
    CACHE STRING
    "Flags for MinSizeRel builds"
)

set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/"
    CACHE INTERNAL
    "Default module path."
)
# Include common cmake modules
include(utils)
# Setup the package version based on git tags
set(PKG_VERSION_GIT_TAG_PREFIX "rdc_pkg_ver")
# Provide git to utilities
find_program(GIT NAMES git)
get_version_from_tag("1.1.0" ${PKG_VERSION_GIT_TAG_PREFIX} GIT)
# VERSION_* variables should be set by get_version_from_tag
message("Package version: ${VERSION_STRING}")

# When cmake -DBUILD_STANDALONE=off, it will not build rdcd and rdci
# which requires the gRPC
option(BUILD_STANDALONE "Build targets for rdci and rdcd" ON)

# When cmake -DBUILD_RUNTIME=off, it will not build the librdc_rocr.so
# which requires the Rocm run time.
option(BUILD_RUNTIME "Build targets for librdc_rocr.so" ON)

# When cmake -DBUILD_PROFILER=off, it will not build the librdc_rocp.so
# which requires the Rocm profiler.
option(BUILD_PROFILER "Build targets for librdc_rocp.so" OFF)

# When cmake -DBUILD_RVS=off, it will not build the librdc_rvs.so
# which requires the RocmValidationSuite
option(BUILD_RVS "Build targets for librdc_rvs.so" ON)

# When cmake -DBUILD_TESTS=off, it will not build RDC tests.
option(BUILD_TESTS "Build test suite" OFF)

# When cmake -DBUILD_EXAMPLES=off, it will not build RDC examples
option(BUILD_EXAMPLES "Build examples" OFF)

# Enable shared libraries for gtest
option(BUILD_SHARED_LIBS "Build shared library (.so) or not." ON)

# Enable address sanitizer
set(ADDRESS_SANITIZER_DEFAULT OFF)
if(DEFINED ENV{ADDRESS_SANITIZER})
    set(ADDRESS_SANITIZER_DEFAULT $ENV{ADDRESS_SANITIZER})
endif()
option(ADDRESS_SANITIZER "Enable address sanitizer" ${ADDRESS_SANITIZER_DEFAULT})

option(CMAKE_VERBOSE_MAKEFILE "Enable verbose output" OFF)

option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile commands for linters and autocompleters" ON)

execute_process(
    COMMAND ${GIT} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(
    "The HASH value of the current HEAD submission in this RDC code repository is : "
    ${GIT_HASH}
)

# Don't print 'Up-to-date' when installing
set(CMAKE_INSTALL_MESSAGE LAZY)

# this must go after some CMAKE_* variables
project(${RDC} VERSION "${VERSION_STRING}" HOMEPAGE_URL "https://github.com/RadeonOpenCompute/rdc")

# Include CMAKE_INSTALL_* variables
# this must go after project()
include(GNUInstallDirs)

set(COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

set(GRPC_ROOT_DEFAULT "/usr")
set(GRPC_ROOT ${GRPC_ROOT_DEFAULT} CACHE PATH "GRPC installation directory.")
set(GRPC_DESIRED_VERSION 1.67.1 CACHE STRING "GRPC desired package version.")

# add package search paths
list(APPEND CMAKE_PREFIX_PATH ${GRPC_ROOT} /usr/local)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/lib64 /usr/lib/x86_64-linux-gnu)

# configure packaging
# cpack version is populated with CMAKE_PROJECT_VERSION implicitly
set(CPACK_PACKAGE_NAME ${RDC_PACKAGE} CACHE INTERNAL "")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc." CACHE STRING "")
set(CPACK_PACKAGE_CONTACT "RDC Support <rdc.support@amd.com>" CACHE STRING "")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Radeon Data Center Tools" CACHE STRING "")
set(CPACK_PACKAGE_DESCRIPTION
    "This package contains the AMD ${CPACK_PACKAGE_DESCRIPTION_SUMMARY}."
    CACHE STRING
    ""
)
set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Default packaging prefix.")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE" CACHE INTERNAL "")
set(CPACK_RPM_PACKAGE_LICENSE "MIT" CACHE INTERNAL "")
set(CPACK_GENERATOR "DEB;RPM" CACHE STRING "Default packaging generators.")
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)
mark_as_advanced(
    CPACK_PACKAGE_VENDOR
    CPACK_PACKAGE_CONTACT
    CPACK_PACKAGE_DESCRIPTION_SUMMARY
    CPACK_PACKAGE_DESCRIPTION
    CPACK_GENERATOR
)

# check if libcap exists
# needed for sys/capabilities.h
find_library(LIB_CAP NAMES cap REQUIRED)

if(BUILD_STANDALONE AND GRPC_ROOT STREQUAL GRPC_ROOT_DEFAULT)
    message(
        WARNING
        "GRPC_ROOT is left default. Cannot install gRPC from default root!
    Please specify -DGRPC_ROOT=<gRPC installation directory>
    Continuing without gRPC install"
    )
endif()

find_package(amd_smi 26.0.0 NAMES amd_smi HINTS ${ROCM_DIR}/lib/cmake CONFIG REQUIRED)

if(NOT EXISTS "${AMD_SMI_INCLUDE_DIR}" OR NOT EXISTS "${AMD_SMI_LIB_DIR}")
    message(
        FATAL_ERROR
        "amd_smi not found in ${AMD_SMI_INCLUDE_DIR}. Please
    make sure amd_smi is installed and present in ${AMD_SMI_INCLUDE_DIR}."
    )
endif()

set(${RDC}_VERSION_MAJOR "${VERSION_MAJOR}")
set(${RDC}_VERSION_MINOR "${VERSION_MINOR}")
set(${RDC}_VERSION_PATCH "0")
set(${RDC}_VERSION_BUILD "0")
set(CPACK_PACKAGE_VERSION ${VERSION_STRING})

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4.0)
    message("Compiler version is " ${CMAKE_CXX_COMPILER_VERSION})
    message(FATAL_ERROR "Require at least gcc-5.4.0")
endif()

message("Build Configuration:")

if(BUILD_STANDALONE)
    message("-----------GRPC ROOT: " ${GRPC_ROOT})
endif()
message("-----------ROCM_DIR : " ${ROCM_DIR})

# this is needed for INSTALL_RPATH "\$ORIGIN" property to work correctly
# please note that because of --enable-new-dtags - this actually sets RUNPATH and not RPATH.
# this means LD_LIBRARY_PATH can override the library locations
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE CACHE BOOL "Set all RPATH to be ORIGIN-based")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Use link path for RPATH")

## Add address sanitizer
if(${ADDRESS_SANITIZER})
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
        message(STATUS "ADDRESS_SANITIZE: CMAKE_CXX_FLAGS=: ${CMAKE_CXX_FLAGS}")
        message(STATUS "ADDRESS_SANITIZE: CMAKE_EXE_LINKER_FLAGS=: ${CMAKE_EXE_LINKER_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -shared-libasan")
        message(STATUS "ADDRESS_SANITIZE: CMAKE_CXX_FLAGS=: ${CMAKE_CXX_FLAGS}")
        message(STATUS "ADDRESS_SANITIZE: CMAKE_EXE_LINKER_FLAGS=: ${CMAKE_EXE_LINKER_FLAGS}")
    endif()
endif()

# Create a configure file to get version info from within library
configure_file(
    "${PROJECT_SOURCE_DIR}/src/${RDC}64Config.in"
    "${PROJECT_SOURCE_DIR}/include/rdc/${RDC}64Config.h"
)

if(BUILD_STANDALONE)
    # Compile .proto files
    file(GLOB PROTOB_DEF_SRC_FILES "protos/*.proto")
    set(PROTOB_SRC_DIR "${PROJECT_SOURCE_DIR}/protos")
    set(PROTOB_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    find_program(PROTOB_CMD NAMES protoc REQUIRED)
    find_program(GRPC_PLUGIN NAMES grpc_cpp_plugin REQUIRED)
    set(GRPC_LIB_DIR "${GRPC_ROOT}/lib")

    set(ENV{LD_LIBRARY_PATH} ${GRPC_LIB_DIR}:${GRPC_LIB_DIR}64:$ENV{LD_LIBRARY_PATH})
    message("LD_LIBRARY_PATH = $ENV{LD_LIBRARY_PATH}")
    foreach(file ${PROTOB_DEF_SRC_FILES})
        execute_process(
            COMMAND ${PROTOB_CMD} --proto_path=${PROTOB_SRC_DIR} --cpp_out=${PROTOB_OUT_DIR} ${file}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE PROTOB_RESULT
            OUTPUT_VARIABLE PROTOB_OUT_VAR
            COMMAND_ERROR_IS_FATAL ANY
        )
        message("protoc command returned: ${PROTOB_RESULT}")

        message("GRPC_PLUGIN=${GRPC_PLUGIN})")
        message("protoc cmd:")
        message("  $ ${PROTOB_CMD} --proto_path=${PROTOB_SRC_DIR}")
        message("    --grpc_out=${PROTOB_OUT_DIR}")
        message("....--plugin=protoc-gen-grpc=\"${GRPC_PLUGIN}\" ${file}")

        execute_process(
            COMMAND
                ${PROTOB_CMD} --proto_path=${PROTOB_SRC_DIR} --grpc_out=${PROTOB_OUT_DIR}
                --plugin=protoc-gen-grpc=${GRPC_PLUGIN} ${file}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE PROTOB_RESULT
            OUTPUT_VARIABLE PROTOB_OUT_VAR
        )
        message("protoc command returned: ${PROTOB_RESULT}")
    endforeach()
endif()

# define all the install component labels to install
set(SERVER_COMPONENT "server")
set(CLIENT_COMPONENT "client")
set(TESTS_COMPONENT "tests")

# Standalone only folders
if(BUILD_STANDALONE)
    # these packages are later used in server and client targets
    find_package(protobuf HINTS ${GRPC_ROOT} CONFIG REQUIRED)
    find_package(gRPC ${GRPC_DESIRED_VERSION} HINTS ${GRPC_ROOT} CONFIG REQUIRED)

    # Don't print grpc install because it floods the terminal
    set(OLD_CMAKE_INSTALL_MESSAGE ${CMAKE_INSTALL_MESSAGE})
    set(CMAKE_INSTALL_MESSAGE NEVER)
    # Only allow installation when GRPC_ROOT is not default
    # The alternative is installing all libraries and binaries from
    # GRPC_ROOT_DEFAULT (/usr), which is incorrect behavior.
    # This is needed because it's very difficult to track all gRPC dependencies
    if(NOT GRPC_ROOT STREQUAL GRPC_ROOT_DEFAULT)
        install(
            DIRECTORY ${GRPC_ROOT}/lib
            USE_SOURCE_PERMISSIONS
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/${RDC}/grpc
            COMPONENT ${SERVER_COMPONENT}
        )
        # In SLES, The libprotobuf is created under lib64 folder, install it as well
        if(EXISTS ${GRPC_ROOT}/lib64)
            install(
                DIRECTORY ${GRPC_ROOT}/lib64/
                USE_SOURCE_PERMISSIONS
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/${RDC}/grpc/lib
                COMPONENT ${SERVER_COMPONENT}
            )
        endif()
        # Also include dev setup for people do not want to build grpc
        file(GLOB install_bin_files "${GRPC_ROOT}/bin/*")
        foreach(ITEM ${install_bin_files})
            list(APPEND bin_files "${ITEM}")
        endforeach()
        install(
            FILES ${bin_files}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/${RDC}/grpc/bin
            COMPONENT ${SERVER_COMPONENT}
        )

        install(
            DIRECTORY ${GRPC_ROOT}/include
            USE_SOURCE_PERMISSIONS
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/${RDC}/grpc
            COMPONENT ${SERVER_COMPONENT}
        )
    endif()
    # Restore printing verbosity
    set(CMAKE_INSTALL_MESSAGE ${OLD_CMAKE_INSTALL_MESSAGE})
    unset(OLD_CMAKE_INSTALL_MESSAGE)

    add_subdirectory("server")
    add_subdirectory("rdci")

    if(BUILD_TESTS)
        add_subdirectory("tests/rdc_tests")
    endif()
endif()

# Folders for both standalone and embedded
add_subdirectory("rdc_libs")

if(BUILD_EXAMPLES)
    add_subdirectory("example")
endif()

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE rdc)

set(CONF_LIBS "librdc_bootstrap.so")
set(CONF_LIB_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

# Create the rdc-config.cmake and rdc-config-version files
configure_file(
    cmake_modules/rdc-config-version.cmake.in
    "${PROJECT_BINARY_DIR}/rdc-config-version.cmake"
    @ONLY
)

include(CMakePackageConfigHelpers)
set(CONFIG_PACKAGE_INSTALL_DIR ${CONF_LIB_DIR}/cmake/${RDC})
configure_package_config_file(
    cmake_modules/rdc-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rdc-config.cmake
    INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
)

# Install the rdc-config.cmake and rdc-config-version.cmake
install(
    FILES "${PROJECT_BINARY_DIR}/rdc-config.cmake" "${PROJECT_BINARY_DIR}/rdc-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${RDC}"
    COMPONENT dev
)

# Install the export set for use with the install-tree
install(EXPORT rdcTargets DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${RDC}" COMPONENT dev)

# License file
install(
    FILES ${CPACK_RESOURCE_FILE_LICENSE}
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    RENAME LICENSE.txt
    COMPONENT dev
)

# Python binding
install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/python_binding
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${RDC}
    COMPONENT ${CLIENT_COMPONENT}
)
install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/authentication
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${RDC}
    COMPONENT ${CLIENT_COMPONENT}
)
install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/example
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}
    COMPONENT dev
)

#Identify between SLES and Centos for setting symlink for rdc.service
#SLES need serice file in /usr/lib/systemd/system/rdc.service
#CENTOS/RHEL Require file in /lib/systemd/system/rdc.service
file(STRINGS /etc/os-release LINUX_DISTRO LIMIT_COUNT 1 REGEX "NAME=")
message("Using Linux Distro: ${LINUX_DISTRO}")
if(LINUX_DISTRO MATCHES "SLES")
    set(DISTRO_ROOT "/usr/lib/systemd/system")
else()
    set(DISTRO_ROOT "/lib/systemd/system")
endif()

# 755 permissions
set(INST_SCR_PERM
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
)
configure_file(
    "${PROJECT_SOURCE_DIR}/src/DEBIAN_postinst.in"
    "${PROJECT_SOURCE_DIR}/DEBIAN/postinst"
    FILE_PERMISSIONS ${INST_SCR_PERM}
    @ONLY
)
configure_file(
    "${PROJECT_SOURCE_DIR}/src/DEBIAN_prerm.in"
    "${PROJECT_SOURCE_DIR}/DEBIAN/prerm"
    FILE_PERMISSIONS ${INST_SCR_PERM}
    @ONLY
)
configure_file(
    "${PROJECT_SOURCE_DIR}/src/RPM_rpm_post.in"
    "${PROJECT_SOURCE_DIR}/RPM/rpm_post"
    FILE_PERMISSIONS ${INST_SCR_PERM}
    @ONLY
)
configure_file(
    "${PROJECT_SOURCE_DIR}/src/RPM_preun.in"
    "${PROJECT_SOURCE_DIR}/RPM/rpm_preun"
    FILE_PERMISSIONS ${INST_SCR_PERM}
    @ONLY
)
configure_file(
    "${PROJECT_SOURCE_DIR}/src/RPM_postun.in"
    "${PROJECT_SOURCE_DIR}/RPM/rpm_postun"
    FILE_PERMISSIONS ${INST_SCR_PERM}
    @ONLY
)
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
    message("Using CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}")
else()
    set(ROCM_LIBPATCH_VERSION)
endif()

# Debian package specific variables
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
    set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()
message("Using CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE}")
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

# Only add extras to RUNTIME package (not tests)
# It makes no sense to modify RDCD service when installing tests
# https://cmake.org/cmake/help/latest/cpack_gen/deb.html#variable:CPACK_DEBIAN_%3CCOMPONENT%3E_PACKAGE_CONTROL_EXTRA
set(CPACK_DEBIAN_RUNTIME_PACKAGE_CONTROL_EXTRA
    "${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst"
    "${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm"
)

option(ROCM_DEP_ROCMCORE "Add debian dependency on rocm-core" OFF)
mark_as_advanced(ROCM_DEP_ROCMCORE)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "amd-smi-lib, libc6")
if(ROCM_DEP_ROCMCORE)
    string(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS ", rocm-core")
endif()
# rdc-tests need rdc
set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS "${CPACK_PACKAGE_NAME}")

# RPM package specific variables
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
    set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()
message("Using CPACK_RPM_PACKAGE_RELEASE ${CPACK_RPM_PACKAGE_RELEASE}")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

## 'dist' breaks manual builds on debian systems due to empty Provides
execute_process(
    COMMAND rpm --eval %{?dist}
    RESULT_VARIABLE PROC_RESULT
    OUTPUT_VARIABLE EVAL_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
    string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()

set(CPACK_RPM_PACKAGE_AUTOREQ 0)
set(CPACK_RPM_PACKAGE_AUTOPROV 0)
set(CPACK_RPM_PACKAGE_REQUIRES "amd-smi-lib")
# rdc-tests need rdc
set(CPACK_RPM_TESTS_PACKAGE_REQUIRES "${CPACK_PACKAGE_NAME}")
list(
    APPEND
    CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "/lib"
    "/usr/sbin"
    "/lib/systemd"
    "/lib/systemd/system"
    "/usr"
    "/opt"
)

# Only add extras to RUNTIME package (not tests)
# It makes no sense to modify RDCD service when installing tests
# https://cmake.org/cmake/help/latest/cpack_gen/rpm.html#variable:CPACK_RPM_PRE_INSTALL_SCRIPT_FILE
set(CPACK_RPM_RUNTIME_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_post")
set(CPACK_RPM_RUNTIME_PRE_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_preun")
set(CPACK_RPM_RUNTIME_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_postun")

# Treat runtime group as package base.
# Without it - the base package would be named 'rdc-runtime'
# resulting in rdc-runtime*.deb and rdc-runtime*.rpm
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")

include(CPack)

# rdc package, no postfix
cpack_add_component_group("runtime")
cpack_add_component(${CLIENT_COMPONENT} GROUP runtime)
cpack_add_component(${SERVER_COMPONENT} GROUP runtime)
cpack_add_component(dev GROUP runtime)

# rdc-tests package, -tests postfix
cpack_add_component_group("tests")
cpack_add_component(${TESTS_COMPONENT} GROUP tests)
