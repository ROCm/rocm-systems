################################################################################
## Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
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
################################################################################

cmake_minimum_required(VERSION 3.16.0)

project(ROCTRACER VERSION 4.1.0)

## Build is not supported on Windows plaform
if(WIN32)
  message(FATAL_ERROR "Windows build is not supported.")
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Werror)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(hsa-runtime64 REQUIRED CONFIG PATHS /opt/rocm/)
find_package(HIP REQUIRED CONFIG PATHS /opt/rocm/)

if(NOT DEFINED LIBRARY_TYPE)
  set(LIBRARY_TYPE SHARED)
endif()

## Set module name
string(TOLOWER "${CMAKE_PROJECT_NAME}" ROCTRACER_NAME)

## Build libraries
include(src/CMakeLists.txt)

## Build tests
if(${LIBRARY_TYPE} STREQUAL SHARED)
add_subdirectory(test ${PROJECT_BINARY_DIR}/test)
endif()

if(${LIBRARY_TYPE} STREQUAL SHARED)

## Installation and packaging
if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
  get_filename_component(DEST_NAME ${CPACK_PACKAGING_INSTALL_PREFIX} NAME)
  get_filename_component(DEST_DIR ${CPACK_PACKAGING_INSTALL_PREFIX} DIRECTORY)
  set(CPACK_PACKAGING_INSTALL_PREFIX ${DEST_DIR})
endif ()
message("-----------Dest-name: ${DEST_NAME}")
message("------Install-prefix: ${CMAKE_INSTALL_PREFIX}")
message("-----------CPACK-dir: ${CPACK_PACKAGING_INSTALL_PREFIX}")

option(FILE_REORG_BACKWARD_COMPATIBILITY "Enable File Reorg with backward compatibility" ON)
if(FILE_REORG_BACKWARD_COMPATIBILITY)
  include(roctracer-backward-compat.cmake)
endif()

## Packaging directives
set(CPACK_GENERATOR "DEB" "RPM" "TGZ" CACHE STRING "CPACK GENERATOR DEB;RPM")
set(ENABLE_LDCONFIG ON CACHE BOOL "Set library links and caches using ldconfig.")
set(CPACK_PACKAGE_NAME "${ROCTRACER_NAME}-dev")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_VERSION_MAJOR ${ROCTRACER_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${ROCTRACER_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${ROCTRACER_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "ROCm Profiler Support <dl.ROCm-Profiler.support@amd.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AMD ROCTRACER library")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
  message("Using CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}")
endif()

# Install license file
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION share/doc/${CPACK_PACKAGE_NAME})

## Debian package specific variables
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
  set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()
message ("Using CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE}")
set (CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

## RPM package specific variables
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
  set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()
message("Using CPACK_RPM_PACKAGE_RELEASE ${CPACK_RPM_PACKAGE_RELEASE}")

set( CPACK_RPM_PACKAGE_LICENSE "MIT")

## 'dist' breaks manual builds on debian systems due to empty Provides
execute_process( COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE)
message("RESULT_VARIABLE ${PROC_RESULT} OUTPUT_VARIABLE: ${EVAL_RESULT}")

if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
  string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
message("CPACK_RPM_PACKAGE_RELEASE: ${CPACK_RPM_PACKAGE_RELEASE}")

include(CPack)

endif()

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
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/inc/roctracer.h
    COMMENT "Generating documentation")

  add_custom_target(doc DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doc/html/index.html
    ${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf)

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf"
    DESTINATION share/doc/roctracer
    RENAME "roctracer.pdf"
    OPTIONAL)

  install(DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/doc/html/"
    DESTINATION share/html/roctracer
    OPTIONAL)

endif()
