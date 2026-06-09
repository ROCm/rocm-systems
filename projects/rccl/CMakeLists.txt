# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.

# CMake version minimum requirements
#==================================================================================================
cmake_minimum_required(VERSION 3.16)

# Determine version from makefiles/version.mk (NCCL-style version variables)
#==================================================================================================
# >>> RCCL-DELTA: RCCL builds as a subproject of rocm-systems, so version.mk is
#     read relative to this project rather than ${CMAKE_SOURCE_DIR} (upstream NCCL).
file(READ makefiles/version.mk VERSION_CONTENT)
# <<< RCCL-DELTA
string(REGEX REPLACE ".*NCCL_MAJOR[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_MAJOR "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_MINOR[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_MINOR "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_PATCH[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_PATCH "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_SUFFIX[ ]*:=[ ]*([a-zA-Z0-9]*).*" "\\1" NCCL_SUFFIX "${VERSION_CONTENT}")
string(REGEX REPLACE ".*PKG_REVISION[ ]*:=[ ]*([0-9]+).*" "\\1" PKG_REVISION "${VERSION_CONTENT}")
math(EXPR NCCL_VERSION_CODE "(${NCCL_MAJOR} * 10000) + (${NCCL_MINOR} * 100) + ${NCCL_PATCH}")

# >>> RCCL-DELTA: src/nccl.h.in is filled via configure_file(), which expands
#     ${NCCL_VERSION}; keep it as an alias of the computed NCCL version code.
set(NCCL_VERSION ${NCCL_VERSION_CODE})
# <<< RCCL-DELTA

# Make version information available to C++ source files
add_compile_definitions(
    NCCL_USE_CMAKE
    NCCL_MAJOR=${NCCL_MAJOR}
    NCCL_MINOR=${NCCL_MINOR}
    NCCL_PATCH=${NCCL_PATCH}
    NCCL_VERSION_CODE=${NCCL_VERSION_CODE}
)

# CMake Toolchain file to define compilers and path to ROCm
#==================================================================================================
if (NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/toolchain-linux.cmake")
  message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# RCCL project
#==================================================================================================
project(rccl CXX)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(NCCL_OS_WINDOWS ON)
    add_definitions(-DNCCL_OS_WINDOWS)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(NCCL_OS_LINUX ON)
    add_definitions(-DNCCL_OS_LINUX)
endif()
set(ENV{NCCL_USE_CMAKE} "1")

# Build options
#==================================================================================================
include(cmake/NcclCompat.cmake)
rccl_define_nccl_compat_options()

option(BUILD_ADDRESS_SANITIZER                 "Enable address sanitizer"                      ${RCCL_DEFAULT_BUILD_ADDRESS_SANITIZER})
option(BUILD_BFD                               "Enable custom backtrace (if bfd.h exists)"     OFF)
option(BUILD_LOCAL_GPU_TARGET_ONLY             "Build only for GPUs detected on this machine"  OFF)
option(BUILD_SHARED_LIBS                       "Build as shared library"                       ON)
option(BUILD_TESTS                             "Build unit test programs"                      OFF)
option(BUILD_EXT_EXAMPLES                      "Build ext-{net,tuner,profiler} example plugins" OFF)
option(COLLTRACE                               "Collective Trace Option"                       ON)
option(DUMP_ASM                                "Disassemble and dump"                          OFF)
option(ENABLE_CODE_COVERAGE                    "Enable code coverage"                          OFF)
option(ENABLE_NPKIT                            "Enable NPKit"                                  OFF)
option(ENABLE_IFC                              "Enable indirect function call"                 OFF)
option(ENABLE_DEVICE_LINKER                    "Build with assembly-extract device linker"     ON)
option(EMIT_LLVM_IR                            "Emit LLVM IR/bitcode for nccl_device APIs"     OFF)
option(GENERATE_SYM_KERNELS                    "Generate symmetric memory kernels"             ON)
option(INSTALL_DEPENDENCIES                    "Force install dependencies"                    OFF)
option(REPORT_KERNEL_RESOURCE_USE              "Append -Rpass-analysis=kernel to CXX flags"    OFF)
option(ROCTX                                   "Enable ROCTX"                                  ${RCCL_DEFAULT_ROCTX})
option(PROFILE                                 "Enable profiling"                              OFF)
option(TIMETRACE                               "Enable time-trace during compilation"          OFF)
option(TRACE                                   "Enable additional tracing"                     OFF)
option(QP_TRACKING                             "Enable per-device QP tracking"                 OFF)
option(FAULT_INJECTION                         "Enable fault injection"                        ON)
option(QUIET_WARNINGS                          "Supress compiler warnings"                     OFF)
option(DWORDX4_INTRINSICS                      "Turn off compilation with dwordx4 intrinsics"  ON)
option(ENABLE_ROCSHMEM                         "Enable rocSHMEM support in RCCL"               OFF)
option(ENABLE_WARP_SPEED                       "Enable WARP_SPEED optimizations"               OFF)

# ROCm / HIP build configuration
#==================================================================================================
# GPU target selection, ROCm toolchain discovery, dependency checks, and HIP
# feature probing are isolated in a dedicated module. Everything in it is
# RCCL-specific with no NCCL counterpart, so keeping it out of this file
# minimizes merge conflicts when syncing upstream NCCL CMake changes. The module
# is include()'d (not add_subdirectory'd), so all variables it sets remain in
# this top-level scope, exactly as before.
include(cmake/RocmBuildConfig.cmake)

# Fill in version templates
#==================================================================================================
## Setup VERSION
set(VERSION_STRING "${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}")
rocm_setup_version(VERSION ${VERSION_STRING})

## Fill in version information for main header file
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include/rccl")
configure_file(src/nccl.h.in ${PROJECT_BINARY_DIR}/include/rccl/rccl.h) # For external linking
configure_file(src/nccl.h.in ${PROJECT_BINARY_DIR}/include/nccl.h)      # Used by some internal files

# Add subdirectories
#==================================================================================================
# Source directory contains library creation
add_subdirectory(src)

# LLVM IR / bitcode for nccl_device APIs (HIP / AMDGPU)
#==================================================================================================
# When EMIT_LLVM_IR=ON, build a standalone LLVM bitcode artifact (librccl_device.bc)
# plus a C wrapper header (nccl_device_wrapper.h) from src/include/nccl_device/.
# The bitcode is generated from the hipify-staged headers that librccl.so itself
# is compiled from, so the IR is guaranteed to match the host library.
if(EMIT_LLVM_IR)
  add_subdirectory(bindings/ir)
  # bindings/ir/ reads headers out of ${PROJECT_BINARY_DIR}/hipify, which is
  # populated by the copy_nccl_device_headers target defined in src/CMakeLists.txt.
  add_dependencies(llvm_ir copy_nccl_device_headers)
  # Convenience aggregate target: builds librccl.so together with the bitcode.
  add_custom_target(rccl_with_ir ALL DEPENDS rccl llvm_ir)
endif()

# Install settings
#==================================================================================================
## Specify install targets
rocm_install_targets(TARGETS rccl)
rocm_install(FILES       ${PROJECT_BINARY_DIR}/include/rccl/rccl.h src/include/plugin/nccl_net.h
             DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rccl)
rocm_install(FILES       src/include/api_trace.h
             DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rccl/amd_detail)
rocm_install(FILES       ${PROJECT_BINARY_DIR}/include/nccl.h
                         ${PROJECT_BINARY_DIR}/include/nccl_device.h
             DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
rocm_install(DIRECTORY   ${PROJECT_BINARY_DIR}/include/nccl_device/
             DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nccl_device)
## Optional: install LLVM bitcode artifacts when EMIT_LLVM_IR is enabled.
## librccl_device.bc, nccl_device_wrapper.h, hipified nccl_device headers,
## and nccl.h (same as rccl.h; nccl_device #includes <nccl.h>) are produced
## before install because rccl_with_ir is part of ALL.
if(EMIT_LLVM_IR)
  rocm_install(FILES       ${PROJECT_BINARY_DIR}/lib/librccl_device.bc
               DESTINATION ${CMAKE_INSTALL_LIBDIR}/rccl)
  rocm_install(FILES       ${PROJECT_BINARY_DIR}/include/nccl_device_wrapper.h
                           ${PROJECT_BINARY_DIR}/include/nccl_device.h
                           ${PROJECT_BINARY_DIR}/include/nccl.h
               DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rccl)
  rocm_install(DIRECTORY   ${PROJECT_BINARY_DIR}/include/nccl_device
               DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rccl)
endif()
## Install tuner config directory for CSV-based tuner
file(COPY tuner DESTINATION ${PROJECT_BINARY_DIR})
rocm_install(DIRECTORY ${PROJECT_BINARY_DIR}/tuner DESTINATION ${CMAKE_INSTALL_DATADIR}/rccl)
rocm_export_targets(
  NAMESPACE roc::
  TARGETS   rccl
  DEPENDS   hip)

## Set package dependencies
if(BUILD_ADDRESS_SANITIZER)
  set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan" )
else()
  set(DEPENDS_HIP_RUNTIME "hip-runtime-amd" )
endif()

rocm_package_add_dependencies(DEPENDS "${DEPENDS_HIP_RUNTIME} >= 4.5.0" "${SMI_LIB_NAME}")
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS "${ROCM_PATH}/lib")
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/opt" "${ROCM_PATH}")

find_file (DEBIAN debian_version debconf.conf PATHS /etc)
if(DEBIAN)
  # Write copyright file
  file(WRITE "${CMAKE_BINARY_DIR}/copyright"
  "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: rccl
Source: https://github.com/ROCm/rccl

Files: *
Copyright: (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
Modifications Copyright (c) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
License: See LICENSE.txt for license information\n")
  rocm_install(FILES "${CMAKE_BINARY_DIR}/copyright" DESTINATION ${CMAKE_INSTALL_DATADIR}/rccl)
  # Write changelog file
  find_program( date_executable date )
  execute_process(COMMAND ${date_executable} -R OUTPUT_VARIABLE TIMESTAMP)
  file(WRITE "${CMAKE_BINARY_DIR}/changelog"
  "rccl (${VERSION_STRING}-1) unstable; urgency=medium

  * Initial release.

 -- RCCL Maintainer <rccl-maintainer@amd.com>  ${TIMESTAMP}\n")
  find_program( gzip_executable gzip )
  execute_process(COMMAND bash "-c" "${gzip_executable} -9 -c -n ${CMAKE_BINARY_DIR}/changelog"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR} OUTPUT_FILE "${CMAKE_BINARY_DIR}/changelog.Debian.gz")
  rocm_install(FILES "${CMAKE_BINARY_DIR}/changelog.Debian.gz" DESTINATION ${CMAKE_INSTALL_DATADIR}/rccl)
  set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "ROCm Communication Collectives Library
  Optimized primitives for collective multi-GPU communication")
endif()

## Install RCCL RAS client (built in src/CMakeLists.txt)
rocm_install(TARGETS rcclras)

if(BUILD_TESTS)
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(tests PACKAGE_NAME unittests)
  if(BUILD_EXT_EXAMPLES)
    add_subdirectory(plugins/net/example)
    add_subdirectory(plugins/tuner/example)
    add_subdirectory(plugins/profiler/example)
  endif()
  add_subdirectory(test)

  if(BUILD_SHARED_LIBS)
    # rccl is defined in src/, so avoid TARGET-signature command here.
    # Use an output-stamp command plus a local custom target dependency.
    set(extract_metadata_stamp "${CMAKE_BINARY_DIR}/extract_metadata.stamp")
    add_custom_command(
      OUTPUT "${extract_metadata_stamp}"
      COMMENT "Extracting metadata from librccl.so"
      COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/extract_metadata.cmake
      COMMAND ${CMAKE_COMMAND} -E touch "${extract_metadata_stamp}"
      DEPENDS rccl
      VERBATIM
    )
    add_custom_target(extract_rccl_metadata ALL DEPENDS "${extract_metadata_stamp}")
  endif()

endif()

rocm_create_package(
  NAME        rccl
  DESCRIPTION "ROCm Communication Collectives Library"
  MAINTAINER  "RCCL Maintainer <rccl-maintainer@amd.com>"
  LDCONFIG)

rccl_add_nccl_compat_package_targets()

if(GENERATE_SYM_KERNELS)
  target_compile_definitions(rccl PRIVATE GENERATE_SYM_KERNELS)
endif()

# Keep librccl outputs in the top-level build dir for existing scripts.
set_target_properties(rccl PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
