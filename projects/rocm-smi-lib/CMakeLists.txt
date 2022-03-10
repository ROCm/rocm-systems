#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 3.6.3)

set(AMD_SMI_LIBS_TARGET "amd_smi_libraries")

set ( BUILD_SHARED_LIBS ON CACHE BOOL "Build shared library (.so) or not.")

## Set default module path if not already set
if(NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
endif()
## Include common cmake modules
include(utils)

if (NOT DEFINED CPACK_RESOURCE_FILE_LICENSE)
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
endif()

set(ROCM_SMI "rocm_smi")
set(ROCM_SMI_COMPONENT "lib${ROCM_SMI}")
set(ROCM_SMI_TARGET "${ROCM_SMI}64")
set(ROCM_SMI_LIB_NAME "lib${ROCM_SMI_TARGET}")

# provide git to utilities
find_program (GIT NAMES git)

## Setup the package version based on git tags.
set(PKG_VERSION_GIT_TAG_PREFIX "rsmi_pkg_ver")
get_package_version_number("1.0.0" ${PKG_VERSION_GIT_TAG_PREFIX} GIT)
message("Package version: ${PKG_VERSION_STR}")
set(${AMD_SMI_LIBS_TARGET}_VERSION_MAJOR "${VERSION_MAJOR}")
set(${AMD_SMI_LIBS_TARGET}_VERSION_MINOR "${VERSION_MINOR}")
set(${AMD_SMI_LIBS_TARGET}_VERSION_PATCH "0")
set(${AMD_SMI_LIBS_TARGET}_VERSION_BUILD "0")

# The following default version values should be updated as appropriate for
# ABI breaks (update MAJOR and MINOR), and ABI/API additions (update MINOR).
# Until ABI stabilizes VERSION_MAJOR will be 0. This should be over-ridden
# by git tags (through "git describe") when they are present.
set(PKG_VERSION_MAJOR 1)
set(PKG_VERSION_MINOR 0)
set(PKG_VERSION_PATCH 0)
set(PKG_VERSION_NUM_COMMIT 0)

## Define default variable and variables for the optional build target
##  rocm_smi_lib-dev
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE STRING "Default installation directory.")
set(COMMON_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Location source code common root.")
set(ROCM_SMI_PACKAGE rocm-smi-lib)

project(${AMD_SMI_LIBS_TARGET})

set(COMMON_PROJ_ROOT ${PROJECT_SOURCE_DIR})

## Verbose output.
set(CMAKE_VERBOSE_MAKEFILE on)

if (CMAKE_COMPILER_IS_GNUCC AND
                         CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4.0)
    message("Compiler version is " ${CMAKE_CXX_COMPILER_VERSION})
    message(FATAL_ERROR "Require at least gcc-5.4.0")
endif()

## Compiler flags
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -Wall -Wextra -fno-rtti -m64 -msse -msse2 -std=c++11 ")
# Security options
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -Wconversion -Wcast-align ")
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS} -Wformat=2 -fno-common -Wstrict-overflow ")
 # Intentionally leave out -Wsign-promo. It causes spurious warnings.
set(CMAKE_CXX_FLAGS
 "${CMAKE_CXX_FLAGS}  -Woverloaded-virtual -Wreorder ")

# Clang does not set the build-id
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set (CMAKE_SHARED_LINKER_FLAGS "-Wl,--build-id=sha1")
endif()

## Address Sanitize Flag
if (${ADDRESS_SANITIZER})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
    set(CMAKE_EXE_LINKER_FLAGS -fsanitize=address)

    if (BUILD_SHARED_LIBS})
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -shared-libsan" )
    else ()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libsan" )
    endif ()
else ()
    ## Security breach mitigation flags
    set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -DFORTIFY_SOURCE=2 -fstack-protector-all -Wcast-align")
    ## More security breach mitigation flags
    set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wl,-z,noexecstack -Wl,-znoexecheap -Wl,-z,relro ")
    set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wtrampolines -Wl,-z,now -fPIE")
endif ()

# Use this instead of above for 32 bit
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")

if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DDEBUG")
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
set(CMN_INC_LIST ${CMN_INC_LIST} "${COMMON_INC_DIR}/rocm_smi.h")
set(CMN_INC_LIST ${CMN_INC_LIST} "${SHR_MUTEX_DIR}/shared_mutex.h")

add_subdirectory("rocm_smi")
add_subdirectory("oam")

option(FILE_REORG_BACKWARD_COMPATIBILITY "Enable File Reorg with backward compatibility" ON)
if(FILE_REORG_BACKWARD_COMPATIBILITY)
  include(rocm_smi-backward-compat.cmake)
endif()
#TODO: Should use GNUInstallDirs to match distro standards
#This need fix in subdirectory CMakefile as well
#include(GNUInstallDirs)
set(LIB_INSTALL_DIR  "lib")
set(INCLUDE_INSTALL_DIR  "include")
set(BIN_INSTALL_DIR  "bin")

include(CMakePackageConfigHelpers)

configure_package_config_file(
    rocm_smi-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rocm_smi-config.cmake
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/rocm_smi
    PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR BIN_INSTALL_DIR
    )

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rocm_smi-config-version.cmake
    VERSION "${AMD_SMI_LIBS_TARGET_VERSION_MAJOR}.${AMD_SMI_LIBS_TARGET_VERSION_MINOR}.${AMD_SMI_LIBS_TARGET_VERSION_PATCH}"
    COMPATIBILITY SameMajorVersion
    )

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rocm_smi-config.cmake ${CMAKE_CURRENT_BINARY_DIR}/rocm_smi-config-version.cmake
    DESTINATION
    lib/cmake/${ROCM_SMI}
    COMPONENT dev
    )

# Create cmake target
# Add all targets to the build-tree export set
export(TARGETS ${ROCM_SMI_TARGET} ${OAM_TARGET}
  FILE "${PROJECT_BINARY_DIR}/rcom_smi_target.cmake")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE rocm_smi)
# Create the rocm_smiConfig.cmake and rocm_smiConfigVersion files
# ... for the build tree
install(EXPORT rocm_smiTargets DESTINATION
  "lib/cmake/${ROCM_SMI}" COMPONENT dev)

#License file
set(CPACK_RPM_PACKAGE_LICENSE "NCSA")
install( FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION share/doc/${ROCM_SMI} RENAME LICENSE.txt)

###########################
# Packaging directives
###########################
set(CPACK_GENERATOR "DEB;RPM"  CACHE STRING "Default packaging generators.")
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm" CACHE STRING "Default packaging prefix.")
set ( ENABLE_LDCONFIG ON CACHE BOOL "Set library links and caches using ldconfig.")
set(CPACK_PACKAGE_NAME ${ROCM_SMI_PACKAGE})
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
set (CPACK_DEBIAN_PACKAGE_PROVIDES "rocm-smi")
set (CPACK_DEBIAN_PACKAGE_SUGGESTS "sudo")
set (CPACK_DEBIAN_PACKAGE_DEPENDS "python3, rocm-core")

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
set (CPACK_RPM_PACKAGE_PROVIDES "rocm-smi")
set (CPACK_RPM_PACKAGE_REQUIRES "python3, rocm-core")


# The line below doesn't currently work; it may be this issue:
# https://bugzilla.redhat.com/show_bug.cgi?id=1811358
# set (CPACK_RPM_PACKAGE_SUGGESTS "sudo")

#Prepare final version for the CPACK use
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${ROCM_VERSION_FOR_PACKAGE}")

## Process the Rpm install/remove scripts to update the CPACK variables
configure_file ( "${CMAKE_CURRENT_SOURCE_DIR}/RPM/post.in" RPM/post @ONLY )
configure_file ( "${CMAKE_CURRENT_SOURCE_DIR}/RPM/postun.in" RPM/postun @ONLY )
set ( CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/post" )
set ( CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/RPM/postun" )

#Set the names now using CPACK utility
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
# Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_PACKAGE_REQUIRES ${CPACK_RPM_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
endif()

include (CPack)
