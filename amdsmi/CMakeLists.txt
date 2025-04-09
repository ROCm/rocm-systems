#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 3.20)

set(AMD_SMI "amd_smi")
set(AMD_SMI_COMPONENT "lib${AMD_SMI}")
set(AMD_SMI_TARGET "${AMD_SMI}")
set(AMD_SMI_LIB_NAME "lib${AMD_SMI_TARGET}")
set(AMD_SMI_LIBS_TARGET "amd_smi_libraries")

set(BUILD_SHARED_LIBS
    ON
    CACHE BOOL "Build shared library (.so) or not.")

set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/"
    CACHE INTERNAL "Default module path.")
## Include common cmake modules
include(utils)
include(help_package)
find_package(PkgConfig)

generic_add_rocm()

# provide git to utilities
find_program(GIT NAMES git)

## Setup the package version based on git tags.
set(PKG_VERSION_GIT_TAG_PREFIX "amdsmi_pkg_ver")
get_package_version_number("25.4.0" ${PKG_VERSION_GIT_TAG_PREFIX} GIT)
message("Package version: ${PKG_VERSION_STR}")
set(${AMD_SMI_LIBS_TARGET}_VERSION_MAJOR "${CPACK_PACKAGE_VERSION_MAJOR}")
set(${AMD_SMI_LIBS_TARGET}_VERSION_MINOR "${CPACK_PACKAGE_VERSION_MINOR}")
set(${AMD_SMI_LIBS_TARGET}_VERSION_PATCH "${CPACK_PACKAGE_VERSION_PATCH}")
set(${AMD_SMI_LIBS_TARGET}_VERSION_BUILD "0")
set(${AMD_SMI_LIBS_TARGET}_VERSION_HASH "${PKG_VERSION_HASH}")

set(${AMD_SMI_LIBS_TARGET}_VERSION_STRING "${${AMD_SMI_LIBS_TARGET}_VERSION_MAJOR}.${${AMD_SMI_LIBS_TARGET}_VERSION_MINOR}.${${AMD_SMI_LIBS_TARGET}_VERSION_PATCH}+${${AMD_SMI_LIBS_TARGET}_VERSION_HASH}")

# Make proper version for appending
# Default Value is 99999
set(ROCM_VERSION_FOR_PACKAGE "99999")
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(ROCM_VERSION_FOR_PACKAGE $ENV{ROCM_LIBPATCH_VERSION})
endif()
#Prepare final version for the CPACK use
set(CPACK_PACKAGE_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${ROCM_VERSION_FOR_PACKAGE}")

# The following default version values should be updated as appropriate for
# ABI breaks (update MAJOR and MINOR), and ABI/API additions (update MINOR).
# Until ABI stabilizes VERSION_MAJOR will be 0. This should be over-ridden
# by git tags (through "git describe") when they are present.
set(PKG_VERSION_MAJOR "${CPACK_PACKAGE_VERSION_MAJOR}")
set(PKG_VERSION_MINOR "${CPACK_PACKAGE_VERSION_MINOR}")
set(PKG_VERSION_PATCH "${CPACK_PACKAGE_VERSION_PATCH}")
set(PKG_VERSION_NUM_COMMIT 0)

set(AMD_SMI_PACKAGE
    amd-smi-lib
    CACHE STRING "")

project(${AMD_SMI_LIBS_TARGET})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

option(BUILD_TESTS "Build test suite" OFF)
option(ENABLE_ASAN_PACKAGING "" OFF)
option(ENABLE_ESMI_LIB "Build ESMI Library" ON)
option(BUILD_EXAMPLES "Build examples" OFF)

include(CMakeDependentOption)
# these options don't work without BUILD_SHARED_LIBS
cmake_dependent_option(BUILD_WRAPPER "Rebuild AMDSMI-wrapper" OFF "BUILD_SHARED_LIBS" OFF)
cmake_dependent_option(BUILD_CLI "Build AMDSMI-CLI and install" ON "BUILD_SHARED_LIBS" OFF)
cmake_dependent_option(BUILD_RUST_WRAPPER "Build rust wrapper and install" OFF "BUILD_SHARED_LIBS" OFF)
cmake_dependent_option(ENABLE_LDCONFIG "Set library links and caches using ldconfig." ON "BUILD_SHARED_LIBS" OFF)

# Set share path here because project name != amd_smi
set(SHARE_INSTALL_PREFIX "${CMAKE_INSTALL_DATAROOTDIR}/${AMD_SMI}" CACHE STRING "Tests and Example install directory")

# Packaging directives
set(CPACK_PACKAGE_NAME
    ${AMD_SMI_PACKAGE}
    CACHE STRING "")

set(CPACK_PACKAGE_CONTACT
    "AMD-SMILib Support <amd-smi.support@amd.com>"
    CACHE STRING "")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "AMD System Management libraries"
    CACHE STRING "")

generic_package()

## Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fno-rtti")
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(CMAKE_CXX_FLAGS
     "${CMAKE_CXX_FLAGS} -m64 -msse -msse2")
endif()
# Security options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion -Wcast-align")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=2 -fno-common -Wstrict-overflow")
# Intentionally leave out -Wsign-promo. It causes spurious warnings.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Woverloaded-virtual -Wreorder")

set(ROCM_SRC_DIR "${PROJECT_SOURCE_DIR}/rocm_smi/src")
set(ROCM_INC_DIR "${PROJECT_SOURCE_DIR}/rocm_smi/include/rocm_smi")
set(SHR_MUTEX_DIR "${PROJECT_SOURCE_DIR}/third_party/shared_mutex")
if(ENABLE_ESMI_LIB)
    # Supported esmi library version tag
    set(current_esmi_tag "esmi_pkg_ver-4.1.2")

    if(NOT EXISTS ${PROJECT_SOURCE_DIR}/esmi_ib_library/src)
        # TODO: use ExternalProject_Add instead or a submodule
        message(STATUS "Adding esmi_ib_library...")
        execute_process(COMMAND git clone --depth=1 -b ${current_esmi_tag} https://github.com/amd/esmi_ib_library.git ${PROJECT_SOURCE_DIR}/esmi_ib_library)
    else()
        message(STATUS "esmi_ib_library already installed, checking version...")

        # Grab latest commit and get the tag
        execute_process(
            COMMAND git rev-list --tags --max-count=1
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/esmi_ib_library
            OUTPUT_VARIABLE latest_commit
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
            COMMAND git describe --tags ${latest_commit} --match "*pkg*"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/esmi_ib_library
            OUTPUT_VARIABLE latest_esmi_tag
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # Update to latest tags if not matched
        if(NOT latest_esmi_tag STREQUAL current_esmi_tag)
            message(STATUS "Updating esmi_ib_library...")
            execute_process(
                COMMAND git clone --depth=1 -b ${current_esmi_tag} https://github.com/amd/esmi_ib_library.git ${PROJECT_SOURCE_DIR}/esmi_ib_library_temp
                RESULT_VARIABLE clone_result
            )
            if(clone_result EQUAL 0)
                file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/esmi_ib_library)
                file(RENAME ${PROJECT_SOURCE_DIR}/esmi_ib_library_temp ${PROJECT_SOURCE_DIR}/esmi_ib_library)
                message(STATUS "Successfully cloned updated esmi_ib_library")
            else()
                file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/esmi_ib_library_temp)
                message(FATAL_ERROR "Failed to clone updated esmi_ib_library")
            endif()
        else()
            message(STATUS "esmi_ib_library is the latest version: ${current_esmi_tag}...")
        endif()
    endif()

    # Make sure to update the amd_hsmp.h file with the corresponding esmi version
    file(COPY "${PROJECT_SOURCE_DIR}/include/amd_smi/impl/amd_hsmp.h"
         DESTINATION "${PROJECT_SOURCE_DIR}/esmi_ib_library/include/asm")

    add_definitions("-DENABLE_ESMI_LIB=1")
    set(ESMI_INC_DIR "${PROJECT_SOURCE_DIR}/esmi_ib_library/include")
    set(ESMI_SRC_DIR "${PROJECT_SOURCE_DIR}/esmi_ib_library/src")
    # esmi has a lot of write-strings warnings - silence them
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings")
endif()

# sets DRM_LIBRARIES and DRM_INCLUDE_DIRS
pkg_check_modules(DRM REQUIRED libdrm)
# sets AMDGPU_DRM_LIBRARIES and AMDGPU_DRM_INCLUDE_DIRS
pkg_check_modules(AMDGPU_DRM REQUIRED libdrm_amdgpu)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party/shared_mutex
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/amd_smi ${ESMI_INC_DIR} ${DRM_INCLUDE_DIRS})

set(CMN_SRC_LIST
    "${ROCM_SRC_DIR}/rocm_smi.cc"
    "${ROCM_SRC_DIR}/rocm_smi_counters.cc"
    "${ROCM_SRC_DIR}/rocm_smi_device.cc"
    "${ROCM_SRC_DIR}/rocm_smi_gpu_metrics.cc"
    "${ROCM_SRC_DIR}/rocm_smi_binary_parser.cc"
    "${ROCM_SRC_DIR}/rocm_smi_io_link.cc"
    "${ROCM_SRC_DIR}/rocm_smi_kfd.cc"
    "${ROCM_SRC_DIR}/rocm_smi_main.cc"
    "${ROCM_SRC_DIR}/rocm_smi_monitor.cc"
    "${ROCM_SRC_DIR}/rocm_smi_power_mon.cc"
    "${ROCM_SRC_DIR}/rocm_smi_utils.cc"
    "${ROCM_SRC_DIR}/rocm_smi_logger.cc"
    "${SHR_MUTEX_DIR}/shared_mutex.cc")

if(ENABLE_ESMI_LIB)
	list(APPEND CMN_SRC_LIST ${ESMI_SRC_DIR}/e_smi.c)
	list(APPEND CMN_SRC_LIST ${ESMI_SRC_DIR}/e_smi_monitor.c)
	list(APPEND CMN_SRC_LIST ${ESMI_SRC_DIR}/e_smi_plat.c)
	list(APPEND CMN_SRC_LIST ${ESMI_SRC_DIR}/e_smi_utils.c)
endif()

set(CMN_INC_LIST
    "${ROCM_INC_DIR}/rocm_smi.h"
    "${ROCM_INC_DIR}/rocm_smi_common.h"
    "${ROCM_INC_DIR}/rocm_smi_counters.h"
    "${ROCM_INC_DIR}/rocm_smi_device.h"
    "${ROCM_INC_DIR}/rocm_smi_gpu_metrics.h"
    "${ROCM_INC_DIR}/rocm_smi_binary_parser.h"
    "${ROCM_INC_DIR}/rocm_smi_exception.h"
    "${ROCM_INC_DIR}/rocm_smi_io_link.h"
    "${ROCM_INC_DIR}/rocm_smi_kfd.h"
    "${ROCM_INC_DIR}/rocm_smi_main.h"
    "${ROCM_INC_DIR}/rocm_smi_monitor.h"
    "${ROCM_INC_DIR}/rocm_smi_power_mon.h"
    "${ROCM_INC_DIR}/rocm_smi_utils.h"
    "${ROCM_INC_DIR}/rocm_smi_logger.h"
    "${SHR_MUTEX_DIR}/shared_mutex.h")

add_subdirectory("rocm_smi")
add_subdirectory("src")

if(BUILD_TESTS)
    set(TESTS_COMPONENT "tests")
    #add_subdirectory("tests/rocm_smi_test")
    add_subdirectory("tests/amd_smi_test")
    add_subdirectory("tests/python_unittest")
endif()

# python interface, CLI, and py-test depend on shared libraries
if(BUILD_SHARED_LIBS)
    add_subdirectory("py-interface")
    if(BUILD_CLI)
        add_subdirectory("amdsmi_cli")
    endif()
    if(BUILD_RUST_WRAPPER)
        add_subdirectory("rust-interface")
    endif()
endif()

if(BUILD_EXAMPLES)
    add_subdirectory("example")
endif()


include(CMakePackageConfigHelpers)

configure_package_config_file(
    amd_smi-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/amd_smi-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${AMD_SMI}
    PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_BINDIR)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/amd_smi-config-version.cmake
    VERSION
    "${CPACK_PACKAGE_VERSION}"
    COMPATIBILITY SameMajorVersion)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/amd_smi-config.cmake ${CMAKE_CURRENT_BINARY_DIR}/amd_smi-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${AMD_SMI}
    COMPONENT dev)

# Create cmake target
# Add all targets to the build-tree export set
export(TARGETS ${AMD_SMI_TARGET} FILE "${PROJECT_BINARY_DIR}/amd_smi_target.cmake")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE amd_smi)

# Create the amd_smiConfig.cmake and amd_smiConfigVersion files
# ... for the build tree
install(
    EXPORT amd_smiTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${AMD_SMI}
    COMPONENT dev)

set(CPACK_RPM_PACKAGE_LICENSE "MIT")
# install license file in share/doc/amd_smi-asan folder
install(
    FILES ${CPACK_RESOURCE_FILE_LICENSE}
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${AMD_SMI}-asan
    RENAME LICENSE.txt
    COMPONENT asan)
# docs are installed into different share directory from tests and examples
install(
    FILES ${CPACK_RESOURCE_FILE_LICENSE}
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${AMD_SMI}
    RENAME LICENSE.txt
    COMPONENT dev)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${AMD_SMI}
    COMPONENT dev)
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/example
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${AMD_SMI}
    COMPONENT dev)

# Make for goamdsmi_shim library
add_subdirectory(goamdsmi_shim)

#Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "python3-argcomplete, libdrm-dev")
set(CPACK_DEBIAN_ASAN_PACKAGE_RECOMMENDS ${CPACK_DEBIAN_PACKAGE_RECOMMENDS})
set(CPACK_DEBIAN_DEV_PACKAGE_RECOMMENDS ${CPACK_DEBIAN_PACKAGE_RECOMMENDS})
set(CPACK_DEBIAN_PACKAGE_DEPENDS "sudo, python3 (>= 3.6.8), python3-pip, python3-setuptools, python3-wheel")
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})

## Process the Debian install/remove scripts to update the CPACK variables
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst.in DEBIAN/postinst @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/amd-smi-lib/prerm.in DEBIAN/prerm @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/amd-smi-lib-tests/prerm.in DEBIAN/prerm @ONLY)
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "DEBIAN/postinst;DEBIAN/prerm")

# Assign control scripts to the AMDSMI Lib & Tests packages
set(CPACK_DEBIAN_DEV_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_BINARY_DIR}/DEBIAN/amd-smi-lib/prerm")
set(CPACK_DEBIAN_TESTS_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_BINARY_DIR}/DEBIAN/amd-smi-lib-tests/prerm")

# RPM package specific variables
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "${CPACK_PACKAGING_INSTALL_PREFIX} ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
#Set rpm distro
if(CPACK_RPM_PACKAGE_RELEASE)
    set(CPACK_RPM_PACKAGE_RELEASE_DIST ON)
endif()
# NOTE: RPM SUGGESTS DO NOT WORK! https://bugzilla.redhat.com/show_bug.cgi?id=1811358
set(CPACK_RPM_PACKAGE_SUGGESTS "python3-argcomplete")
set(CPACK_RPM_DEV_PACKAGE_SUGGESTS ${CPACK_RPM_PACKAGE_SUGGESTS})
set(CPACK_RPM_ASAN_PACKAGE_SUGGESTS ${CPACK_RPM_PACKAGE_SUGGESTS})
# python version gated by rhel8 :(
set(CPACK_RPM_PACKAGE_REQUIRES "sudo, python3 >= 3.6.8, python3-pip, python3-wheel, python3-setuptools")
set(CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_PACKAGE_REQUIRES})
set(CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_PACKAGE_REQUIRES})

# don't terminate if bytecompile of python files fails
set(CPACK_RPM_SPEC_MORE_DEFINE "%define _python_bytecompile_errors_terminate_build 0")
# Cpack converts !/usr/bin/env python3 to /usr/libexec/platform-python in RHEL8.
# prevent the BRP(buildroot policy) script from checking and modifying interpreter directives
string( APPEND CPACK_RPM_SPEC_MORE_DEFINE "\n%undefine __brp_mangle_shebangs")

# Add rocm-core dependency if -DROCM_DEP_ROCMCORE=ON is passed
if(ROCM_DEP_ROCMCORE)
    string(APPEND CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ", rocm-core-asan")
    string(APPEND CPACK_RPM_ASAN_PACKAGE_REQUIRES ", rocm-core-asan")
    string(APPEND CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ", rocm-core")
    string(APPEND CPACK_RPM_DEV_PACKAGE_REQUIRES ", rocm-core")
    string(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS ", rocm-core")
    string(APPEND CPACK_RPM_PACKAGE_REQUIRES ", rocm-core")
endif()

## Enable Component Mode and set component specific flags
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${AMD_SMI_PACKAGE}")
set(CPACK_DEBIAN_TESTS_PACKAGE_NAME "${AMD_SMI_PACKAGE}-tests")
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${AMD_SMI_PACKAGE}-asan")
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_DEV_PACKAGE_NAME "${AMD_SMI_PACKAGE}")
set(CPACK_RPM_TESTS_PACKAGE_NAME "${AMD_SMI_PACKAGE}-tests")
set(CPACK_RPM_ASAN_PACKAGE_NAME "${AMD_SMI_PACKAGE}-asan")
if(ENABLE_ASAN_PACKAGING)
  # ASAN Package requires only asan component with libraries and license file
  set(CPACK_COMPONENTS_ALL asan)
else()
  set(CPACK_COMPONENTS_ALL dev tests )
endif()

# The line below doesn't currently work; it may be this issue:
# https://bugzilla.redhat.com/show_bug.cgi?id=1811358
# set(CPACK_RPM_PACKAGE_SUGGESTS "sudo, libdrm-dev")

## Process the Rpm install/remove scripts to update the CPACK variables
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/RPM/post.in" RPM/post @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/RPM/preun.in" RPM/preun @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/RPM/postun.in" RPM/postun @ONLY)
set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/post")
set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/preun")
set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/postun")

#Set the names now using CPACK utility
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

include(CPack)

generic_package_post()
