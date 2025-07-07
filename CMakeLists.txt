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

cmake_minimum_required(VERSION 3.18.0)

project(roctracer VERSION 4.1.0)

if(${ROCM_PATCH_VERSION})
   set(PROJECT_VERSION_PATCH ${ROCM_PATCH_VERSION})
   set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
endif()

include(GNUInstallDirs)
# set default ROCM_PATH
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory")
endif()

## Build is not supported on Windows plaform
if(WIN32)
  message(FATAL_ERROR "Windows build is not supported.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wno-error=ignored-attributes -Werror)
# To set addition RUNPATH in libraries
# installed in /opt/rocm-ver/lib/roctracer
set(ROCM_APPEND_PRIVLIB_RPATH "$ORIGIN/..")
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

find_package(hsa-runtime64 REQUIRED CONFIG PATHS ${ROCM_PATH})
find_package(HIP REQUIRED CONFIG PATHS ${ROCM_PATH})

if(NOT DEFINED LIBRARY_TYPE)
  set(LIBRARY_TYPE SHARED)
endif()

## Build libraries
add_subdirectory(src)

## Build tests
if(${LIBRARY_TYPE} STREQUAL SHARED)
  add_subdirectory(test)
endif()

## Build Plugins
add_subdirectory(plugin)

if(${LIBRARY_TYPE} STREQUAL SHARED)

## Installation and packaging
if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
  get_filename_component(DEST_NAME ${CPACK_PACKAGING_INSTALL_PREFIX} NAME)
  get_filename_component(DEST_DIR ${CPACK_PACKAGING_INSTALL_PREFIX} DIRECTORY)
  set(CPACK_PACKAGING_INSTALL_PREFIX ${DEST_DIR})
endif()
message("-----------Dest-name: ${DEST_NAME}")
message("------Install-prefix: ${CMAKE_INSTALL_PREFIX}")
message("-----------CPACK-dir: ${CPACK_PACKAGING_INSTALL_PREFIX}")

## Packaging directives
set(CPACK_GENERATOR "DEB" "RPM" "TGZ" CACHE STRING "CPACK GENERATOR DEB;RPM")
set(ENABLE_LDCONFIG ON CACHE BOOL "Set library links and caches using ldconfig.")
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "ROCm Profiler Support <dl.ROCm-Profiler.support@amd.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AMD ROCTRACER library")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
  message("Using CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}")
endif()

## Install license file
install(FILES ${CPACK_RESOURCE_FILE_LICENSE}
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
  COMPONENT runtime)

install(FILES ${CPACK_RESOURCE_FILE_LICENSE}
    DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan
    COMPONENT asan)

## Debian package specific variables
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
  set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()
message("Using CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE}")
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "rocm-core")
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${PROJECT_NAME}-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "${PROJECT_NAME}, hsa-rocr-dev, rocm-core")
set(CPACK_DEBIAN_TESTS_PACKAGE_NAME "${PROJECT_NAME}-tests")
set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS "${PROJECT_NAME}-dev")
# Debian package specific variable for ASAN
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "rocm-core-asan" )

## RPM package specific variables
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
  set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()
message("Using CPACK_RPM_PACKAGE_RELEASE ${CPACK_RPM_PACKAGE_RELEASE}")

set(CPACK_RPM_PACKAGE_LICENSE "MIT")

## 'dist' breaks manual builds on debian systems due to empty Provides
execute_process(COMMAND rpm --eval %{?dist}
                RESULT_VARIABLE PROC_RESULT
                OUTPUT_VARIABLE EVAL_RESULT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
message("RESULT_VARIABLE ${PROC_RESULT} OUTPUT_VARIABLE: ${EVAL_RESULT}")

if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
  string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES "rocm-core")
set(CPACK_RPM_DEV_PACKAGE_NAME "${PROJECT_NAME}-devel")
set(CPACK_RPM_DEV_PACKAGE_REQUIRES "${PROJECT_NAME}, rocm-core")
set(CPACK_RPM_DEV_PACKAGE_PROVIDES "${PROJECT_NAME}-dev")
set(CPACK_RPM_DEV_PACKAGE_OBSOLETES "${PROJECT_NAME}-dev")
set(CPACK_RPM_TESTS_PACKAGE_NAME "${PROJECT_NAME}-tests")
set(CPACK_RPM_TESTS_PACKAGE_REQUIRES "${PROJECT_NAME}-devel, rocm-llvm-devel")
message("CPACK_RPM_PACKAGE_RELEASE: ${CPACK_RPM_PACKAGE_RELEASE}")
# RPM package specific variable for ASAN
set(CPACK_RPM_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan" )

#Disable build id for rocprofiler as its creating transaction error
set ( CPACK_RPM_SPEC_MORE_DEFINE "%define _build_id_links none
                                  %global __strip ${CPACK_STRIP_EXECUTABLE}
                                  %global __objdump ${CPACK_OBJDUMP_EXECUTABLE}
                                  %global __objcopy ${CPACK_OBJCOPY_EXECUTABLE}
                                  %global __readelf ${CPACK_READELF_EXECUTABLE}")

if(NOT ROCM_DEP_ROCMCORE)
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_RUNTIME_PACKAGE_REQUIRES ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES})
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
  string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS})
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
  string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
endif()

if(ENABLE_ASAN_PACKAGING)
  # ASAN Package requires asan component with only libraries and license file
  set(CPACK_COMPONENTS_ALL asan)
else()
  set(CPACK_COMPONENTS_ALL runtime dev tests)
endif()

include(CPack)

cpack_add_component(runtime
  DISPLAY_NAME "Runtime"
  DESCRIPTION "Dynamic libraries for the ROCtracer")

cpack_add_component(dev
  DISPLAY_NAME "Devel"
  DESCRIPTION "Header files and documentation for ROCtracer")

cpack_add_component(tests
  DISPLAY_NAME "Tests"
  DESCRIPTION "Tests for the ROCtracer"
  DEPENDS runtime)

cpack_add_component(asan
  DISPLAY_NAME "ASAN"
  DESCRIPTION "ASAN libraries for the ROCtracer")

endif()

find_package(Doxygen)
if(DOXYGEN_FOUND)
  ## Set input and output files
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  ## Request to configure the file
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doc/html/index.html ${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR}/doc/latex pdf
    MAIN_DEPENDENCY ${DOXYGEN_OUT} ${DOXYGEN_IN}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/inc/roctracer.h ${CMAKE_CURRENT_SOURCE_DIR}/inc/roctracer_plugin.h
    COMMENT "Generating documentation")

  add_custom_target(doc DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doc/html/index.html
    ${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf)

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf"
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    RENAME "roctracer.pdf"
    OPTIONAL
    COMPONENT dev)

  install(DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/doc/html/"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/html/${PROJECT_NAME}
    OPTIONAL
    COMPONENT dev)
endif()
