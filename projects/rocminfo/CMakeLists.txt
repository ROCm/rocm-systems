#
# GCC 4.8 or higher compiler required.
#
#   Required Defines on cmake command line
#
#   1) Set location of ROCR header files (required)
#
#      ROCM_DIR="Root for RocM install"
#
#   2) Set ROCRTST_BLD_TYPE to either "Debug" or "Release".
#      If not set, the default value is "Debug" is bound.
#
#      ROCRTST_BLD_TYPE=Debug or ROCRTST_BLD_TYPE=Release
#
#   3) Set ROCRTST_BLD_BITS to either "32" or "64"
#      If not set, the default value of "64" is bound.
#
#       ROCRTST_BLD_BITS=32 or ROCRTST_BLD_BITS=64
#
#   Building rocminfo
#
#   1) Create build folder e.g. "rocminfo/build" - any name will do
#   2) Cd into build folder
#   3) Run cmake, passing in the above defines, as needed/required:
#       "cmake -DROCM_DIR=<path to rocm root> <other defines if needed> .."
#   4) Run "make"
#
#   Upon a successful build, the executable "rocminfo" will be in the
#   build directory.
#
# Currently support for Windows platform is not present
#

#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 3.6.3)

set(ROCMINFO_EXE "rocminfo")
set(PROJECT_NAME ${ROCMINFO_EXE})
project (${PROJECT_NAME})

include ( GNUInstallDirs )
if(WIN32)
  message("This sample is not supported on Windows platform")
  return()
endif()

# Debug by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

# Generate static package, when BUILD_SHARED_LIBS is set to OFF.
# Default to ON
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
## Set default module path if not already set
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
## Include common cmake modules
include(utils)

#
# Process input variables
#

find_package(hsa-runtime64 1.0 REQUIRED )

if(DEFINED ROCRTST_BLD_TYPE)
  string(TOLOWER "${ROCRTST_BLD_TYPE}" tmp)
  if("${tmp}" STREQUAL release)
    set(CMAKE_BUILD_TYPE "Release")
    set(ISDEBUG 0)
  else()
    set(CMAKE_BUILD_TYPE "Debug")
    set(ISDEBUG 1)
  endif()
endif()

# The following default version values should be updated as appropriate for
# ABI breaks (update MAJOR and MINOR), and ABI/API additions (update MINOR).
# Until ABI stabilizes VERSION_MAJOR will be 0. This should be over-ridden
# by git tags (through "git describe") when they are present.
set(PKG_VERSION_MAJOR 1)
set(PKG_VERSION_MINOR 0)
set(PKG_VERSION_PATCH 0)
set(PKG_VERSION_NUM_COMMIT 0)

################# Determine the library version #########################
## Setup the package version based on git tags.
set(PKG_VERSION_GIT_TAG_PREFIX "rocminfo_pkg_ver")

find_program (GIT NAMES git)

get_package_version_number("1.0.0" ${PKG_VERSION_GIT_TAG_PREFIX} GIT)
# VERSION_* variables should be set by get_version_from_tag
message("Package version: ${PKG_VERSION_STR}")

#
# Print out the build configuration being used:
#
#   Build Src directory
#   Build Binary directory
#   Build Type: Debug Vs Release, 32 Vs 64
#   Compiler Version, etc
#
message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("")


#
# Flag to enable / disable verbose output.
#
SET( CMAKE_VERBOSE_MAKEFILE on )
#
# Compiler pre-processor definitions.
#
# Define MACRO "DEBUG" if build type is "Debug"
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
add_definitions(-DDEBUG)
endif()

add_definitions(-DLITTLEENDIAN_CPU=1)

#
# Linux Compiler options
#
set(ROCMINFO_CXX_FLAGS -std=c++11)
set(ROCMINFO_CXX_FLAGS ${ROCMINFO_CXX_FLAGS} -fexceptions)
set(ROCMINFO_CXX_FLAGS ${ROCMINFO_CXX_FLAGS} -fno-rtti)
set(ROCMINFO_CXX_FLAGS ${ROCMINFO_CXX_FLAGS} -fno-math-errno)
set(ROCMINFO_CXX_FLAGS ${ROCMINFO_CXX_FLAGS} -fno-threadsafe-statics)
set(ROCMINFO_CXX_FLAGS ${ROCMINFO_CXX_FLAGS} -fmerge-all-constants)
set(ROCMINFO_CXX_FLAGS ${ROCMINFO_CXX_FLAGS} -fms-extensions)
set(ROCMINFO_CXX_FLAGS ${ROCMINFO_CXX_FLAGS} -Werror)
set(ROCMINFO_CXX_FLAGS ${ROCMINFO_CXX_FLAGS} -Wall)

#
# Extend the compiler flags for 64-bit builds
#
if((${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64") OR (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64"))
  set(ROCMINFO_CXX_FLAGS ${ROCMINFO_CXX_FLAGS} -m64 -msse -msse2)
endif()

#
# Add compiler flags to include symbol information for debug builds
#
if(ISDEBUG)
  set(ROCMINFO_CXX_FLAGS ${ROCMINFO_CXX_FLAGS} -ggdb -O0)
endif()

###########################
# rocm_agent_enumerator
###########################

configure_file(rocm_agent_enumerator rocm_agent_enumerator COPYONLY)


###########################
# RocR Info
###########################
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ROCMINFO_SOURCES)
add_executable(${ROCMINFO_EXE} ${ROCMINFO_SOURCES})
target_link_libraries(${ROCMINFO_EXE} hsa-runtime64::hsa-runtime64)

target_compile_options(${ROCMINFO_EXE} PRIVATE ${ROCMINFO_CXX_FLAGS})

###########################
# Install directives
###########################
install (
   TARGETS ${ROCMINFO_EXE}
   DESTINATION ${CMAKE_INSTALL_BINDIR} )
install (
   PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/rocm_agent_enumerator
   DESTINATION ${CMAKE_INSTALL_BINDIR} )

###########################
# Packaging directives
###########################
if(BUILD_SHARED_LIBS)
  set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
else()
  set(CPACK_RPM_PACKAGE_NAME "${PROJECT_NAME}-static-devel")
  set(CPACK_DEBIAN_PACKAGE_NAME "${PROJECT_NAME}-static-dev")
endif()
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_VERSION_MAJOR "${PKG_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PKG_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PKG_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "AMD Rocminfo Support <rocminfo.support@amd.com>")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set( CPACK_RPM_PACKAGE_LICENSE "NCSA" )
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Radeon Open Compute (ROCm) Runtime rocminfo tool")

#Install license file
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR})

#Make proper version for appending
#Default Value is 99999, setting it first
set(ROCM_VERSION_FOR_PACKAGE "99999")
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(ROCM_VERSION_FOR_PACKAGE $ENV{ROCM_LIBPATCH_VERSION})
endif()

#Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_DEPENDS "hsa-rocr, pciutils, python3, libc6, libgcc-s1 | libgcc1, libstdc++6")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE ${CPACK_DEBIAN_PACKAGE_HOMEPAGE} CACHE STRING "https://github.com/RadeonOpenCompute/ROCm")
if (DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
   set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
   set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()
if ( ROCM_DEP_ROCMCORE )
  string ( APPEND CPACK_DEBIAN_PACKAGE_DEPENDS ", rocm-core" )
endif()

#RPM package specific variables
execute_process(COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE)
message("RESULT_VARIABLE ${PROC_RESULT} OUTPUT_VARIABLE: ${EVAL_RESULT}")

if(PROC_RESULT EQUAL "0" AND "${EVAL_RESULT}" STREQUAL ".el7")
  # In Centos using parentheses is causing cpack errors.
  # Set the dependencies specifically for centos
  set(CPACK_RPM_PACKAGE_REQUIRES "hsa-rocr, pciutils, python3, glibc, libgcc, libstdc++")
else()
  set(CPACK_RPM_PACKAGE_REQUIRES "hsa-rocr, pciutils, python3, glibc, (libgcc or libgcc_s1), (libstdc++ or libstdc++6)")
endif() # End EVAL_RESULT

if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
  set ( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX} ${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}" )
endif()
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
  set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()
if ( ROCM_DEP_ROCMCORE )
  string ( APPEND CPACK_RPM_PACKAGE_REQUIRES " rocm-core" )
endif()
# Cpack converts !/usr/bin/env python3 to /usr/libexec/platform-python in RHEL8.
# prevent the BRP(buildroot policy) script from checking and modifying interpreter directives
set(CPACK_RPM_SPEC_MORE_DEFINE "%undefine __brp_mangle_shebangs")

#Set rpm distro
if(CPACK_RPM_PACKAGE_RELEASE)
  set(CPACK_RPM_PACKAGE_RELEASE_DIST ON)
endif()

#Prepare final version for the CPACK use
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${ROCM_VERSION_FOR_PACKAGE}")

#Set the names now using CPACK utility
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

if(NOT BUILD_SHARED_LIBS)
  string(REPLACE "hsa-rocr" "hsa-rocr-static-dev" CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
  string(REPLACE "hsa-rocr" "hsa-rocr-static-devel" CPACK_RPM_PACKAGE_REQUIRES ${CPACK_RPM_PACKAGE_REQUIRES})
endif()
include ( CPack )
