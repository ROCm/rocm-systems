## Copyright (c) 2019-2024 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.8)

project(amd-dbgapi VERSION 0.77.4)

include(CheckIncludeFile)
include(GNUInstallDirs)
include(CheckCXXSourceCompiles)

set(CPACK_PACKAGE_NAME rocm-dbgapi)

# Convert the project's name to uppercase and replace '-' with '_'.
string(TOUPPER "${PROJECT_NAME}" AMD_DBGAPI_NAME)
string(REPLACE "-" "_" AMD_DBGAPI_NAME ${AMD_DBGAPI_NAME})

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

# Add an option to enable assertions even in non-debug builds.
if(NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  option(ENABLE_ASSERTIONS "Enable assertions" OFF)
else()
  option(ENABLE_ASSERTIONS "Enable assertions" ON)
endif()

if(ENABLE_ASSERTIONS)
  if(NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    # Add -UNDEBUG to the CFLAGS and CXXFLAGS
    add_compile_options(
      $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-UNDEBUG>)
    # Remove -DNDEBUG from the CFLAGS and CXXFLAGS
    foreach (flags
             CMAKE_C_FLAGS_RELEASE
             CMAKE_C_FLAGS_RELWITHDEBINFO
             CMAKE_C_FLAGS_MINSIZEREL
             CMAKE_CXX_FLAGS_RELEASE
             CMAKE_CXX_FLAGS_RELWITHDEBINFO
             CMAKE_CXX_FLAGS_MINSIZEREL)
      string (REGEX REPLACE "(^| )-D *NDEBUG($| )" " " "${flags}" "${${flags}}")
     endforeach()
  endif()
else()
  if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    add_compile_definitions(NDEBUG)
  endif()
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/amd-dbgapi.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/amd-dbgapi/amd-dbgapi.h @ONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/exportmap.in
  ${CMAKE_CURRENT_BINARY_DIR}/src/exportmap @ONLY)

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/amd-dbgapi.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/share/pkgconfig/amd-dbgapi.pc @ONLY)

add_library(amd-dbgapi SHARED
  src/agent.cpp
  src/architecture.cpp
  src/callbacks.cpp
  src/code_object.cpp
  src/debug.cpp
  src/dispatch.cpp
  src/displaced_stepping.cpp
  src/event.cpp
  src/exception.cpp
  src/initialization.cpp
  src/logging.cpp
  src/memory.cpp
  src/os_driver.cpp
  src/process.cpp
  src/queue.cpp
  src/register.cpp
  src/utils.cpp
  src/versioning.cpp
  src/watchpoint.cpp
  src/wave.cpp
  src/workgroup.cpp)

set_target_properties(amd-dbgapi PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS ON
  CXX_VISIBILITY_PRESET hidden
  OUTPUT_NAME "rocm-dbgapi"
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
  DEFINE_SYMBOL "AMD_DBGAPI_EXPORTS"
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})

target_compile_options(amd-dbgapi PRIVATE
  -fno-rtti -Werror -Wall -Wextra -Wshadow -Wno-attributes) #-pedantic)

target_compile_definitions(amd-dbgapi
  PRIVATE __STDC_LIMIT_MACROS __STDC_CONSTANT_MACROS __STDC_FORMAT_MACROS)

execute_process(
  COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} --print-file-name libbacktrace.a
  RESULT_VARIABLE libbacktrace_check
  OUTPUT_VARIABLE libbacktrace_path
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(${libbacktrace_check} EQUAL 0)
  get_filename_component(BACKTRACE_PREFIX ${libbacktrace_path} DIRECTORY)
  find_path(BACKTRACE_INCLUDE_DIR backtrace-supported.h PATHS ${BACKTRACE_PREFIX}/include)

  if(BACKTRACE_INCLUDE_DIR)
    find_library(BACKTRACE_LIB "backtrace" PATHS ${BACKTRACE_PREFIX})
  endif()

  if(BACKTRACE_LIB)
    target_compile_definitions(amd-dbgapi PRIVATE HAVE_BACKTRACE)
    target_include_directories(amd-dbgapi PRIVATE ${BACKTRACE_INCLUDE_DIR})
    target_link_libraries(amd-dbgapi PRIVATE ${BACKTRACE_LIB})
  endif()
endif()

option(API_TRACING "Enable API tracing" ON)
if(API_TRACING)
  target_compile_definitions(amd-dbgapi PRIVATE WITH_API_TRACING)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_sources(amd-dbgapi PRIVATE src/os_driver_kfd.cpp src/utils_linux.cpp)

  set(CMAKE_REQUIRED_FLAGS_BKP "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++17")
  # For some platforms (PPC64), linux/types.h uses L64, but this code assumes
  # LL64 (i.e.  __u64 should be a long long unsigned, not long unsigned).
  check_cxx_source_compiles(
    "#include <linux/types.h>
     #include <type_traits>
     int main () { static_assert (std::is_same_v<long long unsigned, __u64>); return 0; }"
     LINUX_TYPES_HAS_LL64)

  if(NOT LINUX_TYPES_HAS_LL64)
    # Check if using __SANE_USERSPACE_TYPES__ gives us LL64
    check_cxx_source_compiles(
      "#define __SANE_USERSPACE_TYPES__
       #include <linux/types.h>
       #include <type_traits>
       int main () { static_assert (std::is_same_v<long long unsigned, __u64>); return 0; }"
       LINUX_TYPES_LL64_NEEDS_SANE_USERSPACE_TYPES)
    if(LINUX_TYPES_LL64_NEEDS_SANE_USERSPACE_TYPES)
      target_compile_definitions(amd-dbgapi PRIVATE "__SANE_USERSPACE_TYPES__")
    else()
      message(FATAL_ERROR
              "A LL64 configuration is required for the Linux backend")
    endif()
  endif()
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_BKP}")
else()
  message(FATAL_ERROR "Platform ${CMAKE_SYSTEM_NAME} is not supported")
endif()

find_package(amd_comgr REQUIRED CONFIG
  PATHS
    /opt/rocm/
  PATH_SUFFIXES
    lib/cmake/amd_comgr
)
MESSAGE(STATUS "Code Object Manager found at ${amd_comgr_DIR}.")

if(DEFINED ENV{ROCM_BUILD_ID})
  # ROCM_BUILD_ID is set by the ROCm-CI build environment.
  set(build_info $ENV{ROCM_BUILD_ID})
else()
  string(TIMESTAMP NOW "%Y%m%dT%H%M%S")
  set(build_info developer-build-${NOW})

  if(DEFINED ENV{USER})
    set(build_info ${build_info}-$ENV{USER})
  endif()

  find_package(Git)
  if(GIT_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE build_revision
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
   )
  else()
    message(STATUS "GIT not found")
  endif()

  if(DEFINED build_revision)
    set(build_info ${build_info}-git-${build_revision})
  endif()
endif()

target_link_libraries(amd-dbgapi PRIVATE amd_comgr ${CMAKE_DL_LIBS})

set_source_files_properties(src/versioning.cpp src/initialization.cpp PROPERTIES
  COMPILE_DEFINITIONS "AMD_DBGAPI_VERSION_PATCH=${PROJECT_VERSION_PATCH};AMD_DBGAPI_BUILD_INFO=\"${PROJECT_VERSION}-${build_info}\"")

# We are using the HSA runtime headers, but not the runtime library. Get the HSA runtime
# include directories from the hsa-runtime64::hsa-runtime64 interface.
find_package(hsa-runtime64 REQUIRED CONFIG PATHS "/opt/rocm")
get_property(HSA_RUNTIME_INCLUDE_DIRECTORIES TARGET hsa-runtime64::hsa-runtime64 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

target_include_directories(amd-dbgapi
  PRIVATE
    ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/amd-dbgapi>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(amd-dbgapi
  PRIVATE -Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/src/exportmap -Wl,--no-undefined)

set(AMD_DBGAPI_CONFIG_NAME amd-dbgapi-config.cmake)
set(AMD_DBGAPI_TARGETS_NAME amd-dbgapi-targets.cmake)
set(AMD_DBGAPI_PACKAGE_PREFIX ${CMAKE_INSTALL_LIBDIR}/cmake/amd-dbgapi)

# Generate the build-tree package.
set(AMD_DBGAPI_PREFIX_CODE)
set(AMD_DBGAPI_TARGETS_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/${AMD_DBGAPI_PACKAGE_PREFIX}/${AMD_DBGAPI_TARGETS_NAME}")
export(TARGETS amd-dbgapi
  FILE "${AMD_DBGAPI_PACKAGE_PREFIX}/${AMD_DBGAPI_TARGETS_NAME}")
configure_file("cmake/${AMD_DBGAPI_CONFIG_NAME}.in"
  "${AMD_DBGAPI_PACKAGE_PREFIX}/${AMD_DBGAPI_CONFIG_NAME}"
  @ONLY)

install(TARGETS amd-dbgapi
  EXPORT amd-dbgapi-export
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT dev)

install(TARGETS amd-dbgapi
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT asan)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/include/amd-dbgapi/amd-dbgapi.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd-dbgapi
  COMPONENT dev)

install (FILES
  "${CMAKE_CURRENT_BINARY_DIR}/share/pkgconfig/amd-dbgapi.pc"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/pkgconfig
  COMPONENT dev)

install(FILES
  "LICENSE.txt"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${CPACK_PACKAGE_NAME}
  COMPONENT dev)

install(FILES
  "LICENSE.txt"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${CPACK_PACKAGE_NAME}-asan
  COMPONENT asan)

# Generate the install-tree package.
set(AMD_DBGAPI_PREFIX_CODE "
# Derive absolute install prefix from config file path.
get_filename_component(AMD_DBGAPI_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
string(REGEX REPLACE "/" ";" count "${AMD_DBGAPI_PACKAGE_PREFIX}")
foreach(p ${count})
  set(AMD_DBGAPI_PREFIX_CODE "${AMD_DBGAPI_PREFIX_CODE}
get_filename_component(AMD_DBGAPI_PREFIX \"\${AMD_DBGAPI_PREFIX}\" PATH)")
endforeach()
set(AMD_DBGAPI_TARGETS_PATH "\${AMD_DBGAPI_PREFIX}/${AMD_DBGAPI_PACKAGE_PREFIX}/${AMD_DBGAPI_TARGETS_NAME}")
configure_file("cmake/${AMD_DBGAPI_CONFIG_NAME}.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${AMD_DBGAPI_CONFIG_NAME}.install"
  @ONLY)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${AMD_DBGAPI_CONFIG_NAME}.install"
  DESTINATION "${AMD_DBGAPI_PACKAGE_PREFIX}"
  RENAME "${AMD_DBGAPI_CONFIG_NAME}"
  COMPONENT dev)
install(EXPORT amd-dbgapi-export
  DESTINATION "${AMD_DBGAPI_PACKAGE_PREFIX}"
  FILE "${AMD_DBGAPI_TARGETS_NAME}"
  COMPONENT dev)

# Add packaging directives for amd-dbgapi
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "ROCm Debugger Support <rocm-gdb.support@amd.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Library to provide AMD GPU debugger API")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
  message("Using CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}")
endif()

## Enable Component Mode and set component specific flags
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "rocm-dbgapi")
set(CPACK_RPM_DEV_PACKAGE_NAME "rocm-dbgapi")
set(CPACK_RPM_DEV_PACKAGE_REQUIRES "rocm-core, comgr >= 1.2.0")
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "comgr(>=1.2.0), rocm-core")
# Debian package specific variable for ASAN
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "rocm-dbgapi-asan")
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "comgr-asan(>=1.2.0), rocm-core-asan, libdrm-amdgpu-common | libdrm-common")
# RPM package specific variable for ASAN
set(CPACK_RPM_ASAN_PACKAGE_NAME "rocm-dbgapi-asan" )
set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan, comgr-asan >= 1.2.0, libdrm-amdgpu-common")

# Debian package specific variables
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
  set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()
message("Using CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE}")
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

set(CPACK_DEBIAN_PACKAGE_DEPENDS "comgr(>=1.2.0), rocm-core, libdrm-amdgpu-common | libdrm-common")

if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
  set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()
set(CPACK_RPM_PACKAGE_REQUIRES "rocm-core")
message("Using CPACK_RPM_PACKAGE_RELEASE ${CPACK_RPM_PACKAGE_RELEASE}")

## 'dist' breaks manual builds on debian systems due to empty Provides
execute_process(COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE)
if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
  string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

set(CPACK_RPM_PACKAGE_REQUIRES "comgr >= 1.2.0, libdrm-amdgpu-common")

# Debian package specific variables
#set(CPACK_DEBIAN_PACKAGE_HOMEPAGE
#  "https://github.com/RadeonOpenCompute/")

# RPM package specific variables
if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "${CPACK_PACKAGING_INSTALL_PREFIX}")
endif()

# Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_PACKAGE_REQUIRES ${CPACK_RPM_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
endif()

## set components
if(ENABLE_ASAN_PACKAGING)
  # ASAN Package requires only asan component with libraries and license file
  set(CPACK_COMPONENTS_ALL asan)
else()
  set(CPACK_COMPONENTS_ALL dev)
endif()

if(NOT CPack_CMake_INCLUDED)
  include(CPack)
endif()

cpack_add_component(
  dev
  DISPLAY_NAME "DEV"
  DESCRIPTION "Non ASAN libraries, include files for the ROCM-DBGAPI"
  DEPENDS dev)

cpack_add_component(
  asan
  DISPLAY_NAME "ASAN"
  DESCRIPTION "ASAN libraries for the ROCM-DBGAPI"
  DEPENDS asan)

find_package(Doxygen)
if(DOXYGEN_FOUND)
  # set input and output files
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  # request to configure the file
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doc/html/index.html ${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR}/doc/latex pdf
    MAIN_DEPENDENCY ${DOXYGEN_OUT} ${DOXYGEN_IN}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/amd-dbgapi.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/amd-dbgapi/amd-dbgapi.h
    COMMENT "Generating documentation")

  add_custom_target(doc DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doc/html/index.html
    ${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf)

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${CPACK_PACKAGE_NAME}
    RENAME "amd-dbgapi.pdf"
    OPTIONAL
    COMPONENT dev)

  install(DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/doc/html/"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/html/amd-dbgapi
    OPTIONAL
    COMPONENT dev)

endif()
