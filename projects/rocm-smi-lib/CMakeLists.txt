#
# Minimum version of cmake and C++ required
#
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    CMake ROCm SMI (Library) [root]                ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
cmake_minimum_required(VERSION 3.14)

set(ROCM_SMI_LIBS_TARGET "rocm_smi_libraries")

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared library (.so) or not.")

## Set default module path if not already set
if(NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
endif()
## Include common cmake modules
include(utils)

find_package(PkgConfig)

# Default libdir to "lib", this skips GNUInstallDirs from trying to take a guess if it's unset:
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")

if (NOT DEFINED CPACK_RESOURCE_FILE_LICENSE)
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
endif()

set(ROCM_SMI "rocm_smi")
set(ROCM_SMI_COMPONENT "lib${ROCM_SMI}")
set(ROCM_SMI_TARGET "${ROCM_SMI}64")
set(ROCM_SMI_LIB_NAME "lib${ROCM_SMI_TARGET}")

# Expose project info to IDEs
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile commands for linters and autocompleters" ON)
option(BUILD_TESTS "Build test suite" OFF)
set(SHARE_INSTALL_PREFIX
    "share/${ROCM_SMI}"
    CACHE STRING "Tests and Example install directory")

# provide git to utilities
find_program (GIT NAMES git)


## Setup the package version based on git tags.
set(PKG_VERSION_GIT_TAG_PREFIX "rsmi_pkg_ver")
get_package_version_number("7.6.0" ${PKG_VERSION_GIT_TAG_PREFIX} GIT)
message("Package version: ${PKG_VERSION_STR}")
set(${ROCM_SMI_LIBS_TARGET}_VERSION_MAJOR "${CPACK_PACKAGE_VERSION_MAJOR}")
set(${ROCM_SMI_LIBS_TARGET}_VERSION_MINOR "${CPACK_PACKAGE_VERSION_MINOR}")
set(${ROCM_SMI_LIBS_TARGET}_VERSION_PATCH "${CPACK_PACKAGE_VERSION_PATCH}")
set(${ROCM_SMI_LIBS_TARGET}_VERSION_BUILD "0")
set(${ROCM_SMI_LIBS_TARGET}_VERSION_HASH "${PKG_VERSION_HASH}")

set(${ROCM_SMI_LIBS_TARGET}_VERSION_STRING "${${ROCM_SMI_LIBS_TARGET}_VERSION_MAJOR}.${${ROCM_SMI_LIBS_TARGET}_VERSION_MINOR}.${${ROCM_SMI_LIBS_TARGET}_VERSION_PATCH}+${${ROCM_SMI_LIBS_TARGET}_VERSION_HASH}")

# The following default version values should be updated as appropriate for
# ABI breaks (update MAJOR and MINOR), and ABI/API additions (update MINOR).
# Until ABI stabilizes VERSION_MAJOR will be 0. This should be over-ridden
# by git tags (through "git describe") when they are present.
set(PKG_VERSION_MAJOR "${VERSION_MAJOR}")
set(PKG_VERSION_MINOR "${VERSION_MINOR}")
set(PKG_VERSION_PATCH "${CPACK_PACKAGE_VERSION_PATCH}")
set(PKG_VERSION_NUM_COMMIT 0)

## Define default variable and variables for the optional build target
##  rocm_smi_lib
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE STRING "Default installation directory.")
set(COMMON_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Location source code common root.")
set(ROCM_SMI_PACKAGE rocm-smi-lib)

project(${ROCM_SMI_LIBS_TARGET})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
set(COMMON_PROJ_ROOT ${PROJECT_SOURCE_DIR})

if (CMAKE_COMPILER_IS_GNUCC AND
                         CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4.0)
    message("Compiler version is " ${CMAKE_CXX_COMPILER_VERSION})
    message(FATAL_ERROR "Require at least gcc-5.4.0")
endif()

## Compiler flags
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -Wall -Wextra -fno-rtti")
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(CMAKE_CXX_FLAGS
     "${CMAKE_CXX_FLAGS} -m64 -msse -msse2")
endif()

# Security options
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -Wconversion -Wcast-align ")
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -Wformat=2 -fno-common -Wstrict-overflow ")
 # Intentionally leave out -Wsign-promo. It causes spurious warnings.
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -Woverloaded-virtual -Wreorder ")

# Clang does not set the build-id
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--build-id=sha1")
endif()

# Use this instead of above for 32 bit
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")

if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DDEBUG")
endif ()

## Address Sanitize Flag
if (${ADDRESS_SANITIZER})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS -fsanitize=address)

    if (BUILD_SHARED_LIBS)
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -shared-libasan")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -shared-libasan")
        endif()
    else ()
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libsan")
        else()
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libasan")
        endif()
    endif ()
else ()
    ## Security breach mitigation flags
    set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -DFORTIFY_SOURCE=2 -fstack-protector-all -Wcast-align")
    ## More security breach mitigation flags
    set(HARDENING_LDFLAGS
      "${HARDENING_LDFLAGS} -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${HARDENING_LDFLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${HARDENING_LDFLAGS}")

    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-Wtrampolines" CXX_SUPPORTS_WTRAMPOLINES)
    if (CXX_SUPPORTS_WTRAMPOLINES)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wtrampolines")
    endif ()
endif ()

set(COMMON_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(COMMON_INC_DIR "${PROJECT_SOURCE_DIR}/include/rocm_smi")
set(SHR_MUTEX_DIR "${PROJECT_SOURCE_DIR}/third_party/shared_mutex")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/shared_mutex)

set(CMN_SRC_LIST "${COMMON_SRC_DIR}/rocm_smi_device.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${COMMON_SRC_DIR}/rocm_smi_main.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${COMMON_SRC_DIR}/rocm_smi_monitor.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${COMMON_SRC_DIR}/rocm_smi_power_mon.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${COMMON_SRC_DIR}/rocm_smi_utils.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${COMMON_SRC_DIR}/rocm_smi_counters.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${COMMON_SRC_DIR}/rocm_smi_kfd.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${COMMON_SRC_DIR}/rocm_smi_io_link.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${COMMON_SRC_DIR}/rocm_smi_gpu_metrics.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${COMMON_SRC_DIR}/rocm_smi.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${COMMON_SRC_DIR}/rocm_smi_logger.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${COMMON_SRC_DIR}/rocm_smi_properties.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${COMMON_SRC_DIR}/rocm_smi_lib_loader.cc")
set(CMN_SRC_LIST ${CMN_SRC_LIST} "${SHR_MUTEX_DIR}/shared_mutex.cc")

set(CMN_INC_LIST "${COMMON_INC_DIR}/rocm_smi_device.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_main.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_monitor.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_power_mon.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_utils.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_common.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_exception.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_counters.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_kfd.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_io_link.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_gpu_metrics.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_logger.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_properties.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi_lib_loader.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${SHR_MUTEX_DIR}/shared_mutex.h")

## set components
if(ENABLE_ASAN_PACKAGING)
  # ASAN Package requires only asan component with libraries and license file
  set(CPACK_COMPONENTS_ALL asan)
else()
  set(CPACK_COMPONENTS_ALL dev tests)
endif()

# Enable Component Mode & Install Settings
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)

# Add dependencies
# no rocm-core
# python doesn't need to be asan
set(CPACK_DEBIAN_PACKAGE_DEPENDS "python3")
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}")
# Only add dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON is given
if(ROCM_DEP_ROCMCORE)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, rocm-core")
    # rocm-core needs to be asan
    # override original variable because CPACK_DEBIAN_PACKAGE_DEPENDS changed
    set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}-asan")
endif()
# carefully reuse DEB's "DEPENDS" for RPM's "REQUIRES"
set(CPACK_RPM_PACKAGE_REQUIRES "${CPACK_DEBIAN_PACKAGE_DEPENDS}")
set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS}")

#Component Specific Configuration/Flags
set(CPACK_DEBIAN_DEV_PACKAGE_NAME ${ROCM_SMI_PACKAGE})
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME ${ROCM_SMI_PACKAGE}-asan)
set(CPACK_DEBIAN_TESTS_PACKAGE_NAME ${ROCM_SMI_PACKAGE}-tests)
set(CPACK_DEBIAN_STATIC_PACKAGE_NAME ${ROCM_SMI_PACKAGE}-static-dev)
set(CPACK_RPM_DEV_PACKAGE_NAME ${ROCM_SMI_PACKAGE})
set(CPACK_RPM_ASAN_PACKAGE_NAME ${ROCM_SMI_PACKAGE}-asan)
set(CPACK_RPM_TESTS_PACKAGE_NAME ${ROCM_SMI_PACKAGE}-tests)
set(CPACK_RPM_STATIC_PACKAGE_NAME ${ROCM_SMI_PACKAGE}-static-devel)

add_subdirectory("rocm_smi")
add_subdirectory("oam")

option(FILE_REORG_BACKWARD_COMPATIBILITY "Enable File Reorg with backward compatibility" OFF)

# Add tests
if(BUILD_TESTS)
    set(TESTS_COMPONENT "tests")
    add_subdirectory("tests/rocm_smi_test")
endif()

if(FILE_REORG_BACKWARD_COMPATIBILITY)
# To enable/disable #error in wrapper header files
  if(NOT DEFINED ROCM_HEADER_WRAPPER_WERROR)
      if(DEFINED ENV{ROCM_HEADER_WRAPPER_WERROR})
          set(ROCM_HEADER_WRAPPER_WERROR "$ENV{ROCM_HEADER_WRAPPER_WERROR}"
                  CACHE STRING "Header wrapper warnings as errors.")
      else()
          set(ROCM_HEADER_WRAPPER_WERROR "OFF" CACHE STRING "Header wrapper warnings as errors.")
      endif()
  endif()
  if(ROCM_HEADER_WRAPPER_WERROR)
      set(deprecated_error 1)
  else()
      set(deprecated_error 0)
  endif()

  include(rocm_smi-backward-compat.cmake)
endif()

include(CMakePackageConfigHelpers)

set(LIB_INSTALL_DIR  "${CMAKE_INSTALL_LIBDIR}")
set(INCLUDE_INSTALL_DIR  "${CMAKE_INSTALL_INCLUDEDIR}")
set(BIN_INSTALL_DIR  "${CMAKE_INSTALL_BINDIR}")


configure_package_config_file(
    rocm_smi-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rocm_smi-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocm_smi
    PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR BIN_INSTALL_DIR
    )

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rocm_smi-config-version.cmake
    VERSION "${PKG_VERSION_STR}"
    COMPATIBILITY SameMajorVersion
    )

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rocm_smi-config.cmake ${CMAKE_CURRENT_BINARY_DIR}/rocm_smi-config-version.cmake
    DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${ROCM_SMI}
    COMPONENT dev
    )

# Create cmake target
# Add all targets to the build-tree export set
export(TARGETS ${ROCM_SMI_TARGET} ${OAM_TARGET}
  FILE "${PROJECT_BINARY_DIR}/rocm_smi_target.cmake")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE rocm_smi)

# Create the rocm_smiConfig.cmake and rocm_smiConfigVersion files
# ... for the build tree
install(EXPORT rocm_smiTargets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${ROCM_SMI}"
    COMPONENT dev)

#License file
set(CPACK_RPM_PACKAGE_LICENSE "NCSA")
# install license file in share/doc/rocm_smi-asan folder
if( ENABLE_ASAN_PACKAGING )
  install(FILES ${CPACK_RESOURCE_FILE_LICENSE}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${ROCM_SMI}-asan RENAME LICENSE.txt
        COMPONENT asan)
endif()
install( FILES ${CPACK_RESOURCE_FILE_LICENSE}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${ROCM_SMI} RENAME LICENSE.txt
        COMPONENT dev)

###########################
# Packaging directives
###########################
set(CPACK_GENERATOR "DEB;RPM"  CACHE STRING "Default packaging generators.")
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm" CACHE STRING "Default packaging prefix.")
set ( ENABLE_LDCONFIG ON CACHE BOOL "Set library links and caches using ldconfig.")
if( ENABLE_ASAN_PACKAGING )
  set(CPACK_PACKAGE_NAME ${ROCM_SMI_PACKAGE}-asan)
else()
  set(CPACK_PACKAGE_NAME ${ROCM_SMI_PACKAGE})
endif()
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_CONTACT "RocmSMILib Support <rocm-smi.support@amd.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AMD System Management libraries")

#Make proper version for appending
#Default Value is 99999, setting it first
set(ROCM_VERSION_FOR_PACKAGE "99999")
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(ROCM_VERSION_FOR_PACKAGE $ENV{ROCM_LIBPATCH_VERSION})
endif()

#Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE ${CPACK_DEBIAN_PACKAGE_HOMEPAGE} CACHE STRING "https://github.com/RadeonOpenCompute/ROCm")
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
   set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
   set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()
set(CPACK_DEBIAN_PACKAGE_SUGGESTS "sudo, libdrm-dev, libdrm-amdgpu-dev")
set(CPACK_RPM_PACKAGE_SUGGESTS "sudo, libdrm-dev, libdrm-amdgpu-dev")

## Process the Debian install/remove scripts to update the CPACK variables
configure_file ( ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst.in DEBIAN/postinst @ONLY )
configure_file ( ${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm.in DEBIAN/prerm @ONLY )
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "DEBIAN/postinst;DEBIAN/prerm")

# RPM package specific variables
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX} ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}" )
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
  set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()
#Set rpm distro
if(CPACK_RPM_PACKAGE_RELEASE)
  set(CPACK_RPM_PACKAGE_RELEASE_DIST ON)
endif()

# don't terminate if bytecompile of python files fails
set(CPACK_RPM_SPEC_MORE_DEFINE "%define _python_bytecompile_errors_terminate_build 0")
# Cpack converts !/usr/bin/env python3 to /usr/libexec/platform-python in RHEL8.
# prevent the BRP(buildroot policy) script from checking and modifying interpreter directives
string( APPEND CPACK_RPM_SPEC_MORE_DEFINE "\n%undefine __brp_mangle_shebangs")

# The line below doesn't currently work; it may be this issue:
# https://bugzilla.redhat.com/show_bug.cgi?id=1811358
# set (CPACK_RPM_PACKAGE_SUGGESTS "sudo")

#Prepare final version for the CPACK use
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${ROCM_VERSION_FOR_PACKAGE}")

## Process the Rpm install/remove scripts to update the CPACK variables
configure_file ( "${CMAKE_CURRENT_SOURCE_DIR}/RPM/post.in" RPM/post @ONLY )
configure_file ( "${CMAKE_CURRENT_SOURCE_DIR}/RPM/preun.in" RPM/preun @ONLY )
configure_file ( "${CMAKE_CURRENT_SOURCE_DIR}/RPM/postun.in" RPM/postun @ONLY )
set ( CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/post" )
set ( CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/preun" )
set ( CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/postun" )

#Set the names now using CPACK utility
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

include (CPack)

cpack_add_component(
  dev
  DISPLAY_NAME "Development"
  DESCRIPTION "Development needed header files for ROCM-SMI"
  DEPENDS dev)

cpack_add_component(
  asan
  DISPLAY_NAME "ASAN"
  DESCRIPTION "ASAN libraries for the ROCM-SMI"
  DEPENDS asan)

cpack_add_component_group("tests")
cpack_add_component(tests GROUP tests)
if(NOT BUILD_SHARED_LIBS)
  cpack_add_component_group("static")
  cpack_add_component(dev GROUP static)
endif()

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    CMake ROCm SMI (Library) [root] END            ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
