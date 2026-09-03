###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Set root directory for ROCm Systems shared files
set(ROCM_SYSTEMS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

###############################################################################
# AVOID IN SOURCE BUILD
###############################################################################
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND
   CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(MSG "")
  message(STATUS "Warning! Building from the source directory is not recommended")
  message(STATUS "If unintended, please remove 'CMakeCache.txt' and 'CMakeFiles'")
  message(STATUS "and build from a separate directory")
  message(FATAL_ERROR "In-source build")
endif()

###############################################################################
# CONFIGURATION OPTIONS
###############################################################################

option(PROFILE "Enable statistics and timing support" OFF)
option(USE_RO "Enable RO conduit" OFF)
option(USE_VERBS "Enable backend-neutral verbs networking layer (src/net)" OFF)
option(USE_IPC "Enable IPC support (using HIP)" ON)
option(USE_GDA "Enable GDA conduit" OFF)
option(USE_SDMA "Enable SDMA transport (MI300X+)" OFF)
option(USE_THREADS "Enable workgroup threads to share network queues" OFF)
option(USE_WF_COAL "Enable wavefront message coalescing" OFF)
option(USE_FUNC_CALL "Force compiler to use function calls on library API" OFF)
option(USE_SHARED_CTX "Request support for shared ctx between WG" OFF)
option(USE_SINGLE_NODE "Enable single node support only." OFF)
option(USE_HDP_FLUSH "Force flush the HDP cache." OFF)
option(USE_HDP_FLUSH_HOST_SIDE "Use a polling thread to flush the HDP cache on the host." OFF)
option(USE_ROCPROFILER_REGISTER "Register rocSHMEM API tables with rocprofiler-register so rocprofiler-sdk tools can trace host-stream APIs." ON)

option(BUILD_FUNCTIONAL_TESTS "Build the functional tests (uses MPI if available, otherwise SLR)" OFF)
option(BUILD_EXAMPLES "Build the examples" OFF)
option(BUILD_UNIT_TESTS "Build the unit tests (Requires MPI)" OFF)
option(BUILD_TESTS_ONLY "Build only tests. Used to link against rocSHMEM in a ROCm Release" OFF)
option(BUILD_TOOLS "Build binary tools (e.g., rocshmem_info)" ON)
option(BUILD_CTESTS "Register tests with CTest for tier-based execution" ON)

option(BUILD_LOCAL_GPU_TARGET_ONLY "Build only for GPUs detected on this machine" OFF)
option(BUILD_CODE_COVERAGE "Build with code coverage flags (gcc only)" OFF)
option(BUILD_DEBUG_TRACE_HOST "Compile in host-side trace logging (LOG_API/TRACE)" OFF)
option(BUILD_DEBUG_TRACE_DEVICE "Compile in device-side trace logging (LOGD_API/TRACE)" OFF)
option(BUILD_DEBUG_DEVICE "Compile in device-side logging (LOGD_WARN/INFO and GDA error checking)" OFF)

option(GDA_IONIC "Build for AMD Pensando IONIC RDMA provider" OFF)
option(GDA_BNXT "Build for Broadcom RDMA provider" OFF)
option(GDA_MLX5 "Build for Mellanox MLX5 RDMA provider" OFF)
option(ASAN "Enable AddressSanitizer" ${ASAN})

set(USE_EXTERNAL_MPI AUTO CACHE STRING "Link with an external MPI (required if used MPI is ABI incompatible with Open MPI v5)")
set_property(CACHE USE_EXTERNAL_MPI PROPERTY STRINGS AUTO ON OFF)

###############################################################################
# PROJECT
###############################################################################
include(${CMAKE_SOURCE_DIR}/cmake/setup_project.cmake)

## Setup VERSION
set(VERSION_STRING 3.7.0)
message(STATUS "rocSHMEM Version: " "${VERSION_STRING}")

project(rocshmem VERSION ${VERSION_STRING} LANGUAGES CXX)

# Must run after project(... LANGUAGES CXX) so CMAKE_CXX_COMPILER is populated;
# querying it any earlier silently returns an empty path here, which then
# corrupts the -Wl,-rpath, link flag added for ASAN builds below.
if (ASAN)
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libclang_rt.asan-x86_64.so
    OUTPUT_VARIABLE ASAN_RUNTIME_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  get_filename_component(ASAN_RUNTIME_DIR "${ASAN_RUNTIME_PATH}" DIRECTORY)
  message(STATUS "ASAN runtime directory: ${ASAN_RUNTIME_DIR}")
endif()

find_package(ROCmCMakeBuildTools)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMCheckTargetIds)

rocm_setup_version(VERSION ${VERSION_STRING})

#############################################################################
# DETECT POD CAPABILITY
#############################################################################
# Try to find AMD SMI library
find_package(amd_smi QUIET)

set(HAVE_AMDSMI_GPU_FABRIC_INFO FALSE)

if(TARGET amd_smi)
  include(CheckLibraryExists)
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)

  # Check if AMD SMI library has fabric info support
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES amd_smi)
  check_library_exists(amd_smi amdsmi_get_gpu_fabric_info "" HAVE_AMDSMI_FABRIC_FUNC)
  cmake_pop_check_state()

  find_path(_hip_include_dir
    hip/hip_runtime_api.h
    HINTS "${ROCM_PATH}"
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
  )
  cmake_push_check_state(RESET)
  if(_hip_include_dir)
    set(CMAKE_REQUIRED_INCLUDES ${_hip_include_dir})
  endif()
  set(CMAKE_REQUIRED_DEFINITIONS -D__HIP_PLATFORM_AMD__)
  check_cxx_source_compiles("
    #include <hip/hip_runtime_api.h>
    int main() {
      hipMemFabricHandle_t handle;
      (void)handle;
      return 0;
    }
  " HAVE_HIP_FABRIC_HANDLE)
  cmake_pop_check_state()

  # Both conditions must be met for full fabric support
  if(HAVE_AMDSMI_FABRIC_FUNC AND HAVE_HIP_FABRIC_HANDLE)
    set(HAVE_AMDSMI_GPU_FABRIC_INFO TRUE)
    message(STATUS "AMD SMI library found with fabric info support - enabling pod-based IPC capability detection")
  else()
    if(NOT HAVE_AMDSMI_FABRIC_FUNC)
      message(STATUS "AMD SMI library found but lacks amdsmi_get_gpu_fabric_info - disabling pod-based IPC capability detection")
    endif()
    if(NOT HAVE_HIP_FABRIC_HANDLE)
      message(STATUS "HIP runtime lacks hipMemFabricHandle_t support (requires ROCm 7.14+) - disabling fabric handle features")
    endif()
  endif()
else()
  message(STATUS "AMD SMI library not found - disabling pod-based IPC capability detection")
endif()

#############################################################################
# SET GPU ARCHITECTURES
#############################################################################
include(cmake/rocm_local_targets.cmake)

set(DEFAULT_GPUS
    gfx90a;
    gfx1100;
    gfx1201;
    gfx942)

if(${ROCM_MAJOR_VERSION} GREATER 6)
  list(APPEND DEFAULT_GPUS gfx950)
endif()

if(${ROCM_MAJOR_VERSION} GREATER 7 OR (${ROCM_MAJOR_VERSION} EQUAL 7 AND ${ROCM_MINOR_VERSION} GREATER_EQUAL 14))
  list(APPEND DEFAULT_GPUS gfx1250)
endif()

if($ENV{BUILD_LOCAL_GPU_TARGET_ONLY})
  set(BUILD_LOCAL_GPU_TARGET_ONLY ON)
endif()

if (BUILD_LOCAL_GPU_TARGET_ONLY)
  message(STATUS "Building only for local GPU target")
  if (COMMAND rocm_local_targets)
    rocm_local_targets(DEFAULT_GPUS)
  else()
    message(WARNING "Unable to determine local GPU targets. Falling back to default GPUs.")
  endif()
endif()

set(DEFAULT_GPU_TARGETS "${DEFAULT_GPUS}" CACHE STRING
    "Target default GPUs if GPU_TARGETS is not defined.")

# When the user supplies GPU_TARGETS:
#   - Normalise comma/semicolon separators
#   - Dedupe on the exact string, since the same base arch may legitimately
#     appear more than once with different feature suffixes (e.g.
#     gfx90a:xnack+;gfx90a:xnack-) to fat-binary both variants
# rocm_check_target_ids validates each entry with a real compiler-flag check
# (-xhip --offload-arch=<entry>), so feature-suffixed entries like
# gfx950:sramecc+:xnack- are validated natively and echoed back verbatim on
# success — no separate bare/full bookkeeping is needed.
if(GPU_TARGETS)
  string(REPLACE "," ";" GPU_TARGETS "${GPU_TARGETS}")
  list(REMOVE_DUPLICATES GPU_TARGETS)
endif()

if(COMMAND rocm_check_target_ids)
  if(GPU_TARGETS)
    message(STATUS "Checking for ROCm support for GPU targets: " "${GPU_TARGETS}")
    rocm_check_target_ids(SUPPORTED_GPUS TARGETS ${GPU_TARGETS})
  else()
    message(STATUS "Checking for ROCm support for GPU targets: " "${DEFAULT_GPU_TARGETS}")
    rocm_check_target_ids(SUPPORTED_GPUS TARGETS ${DEFAULT_GPU_TARGETS})
  endif()
else()
  message(WARNING "Unable to check for supported GPU targets.")
  if(GPU_TARGETS)
    set(SUPPORTED_GPUS ${GPU_TARGETS})
  else()
    set(SUPPORTED_GPUS ${DEFAULT_GPU_TARGETS})
  endif()
endif()

# GPU_TARGETS_FULL: validated arch strings with feature suffixes intact.
# Consumed by DeviceBitcode.cmake to embed correct amdhsa.target metadata in
# the device bitcode — HIP error 209 occurs if the suffix in the bitcode
# metadata does not match the active feature configuration of the GPU.
# Explicitly cleared when GPU_TARGETS is not set so a stale cached value from a
# previous -DGPU_TARGETS=... configure run cannot persist into the defaults path.
if(GPU_TARGETS)
  set(GPU_TARGETS_FULL "${SUPPORTED_GPUS}" CACHE STRING
    "Validated GPU arch strings with feature suffixes (consumed by DeviceBitcode.cmake)" FORCE)
else()
  unset(GPU_TARGETS_FULL CACHE)
endif()

set(GPU_TARGETS "${SUPPORTED_GPUS}" CACHE STRING "GPU architectures to compile for" FORCE)

message(STATUS "Compiling for ${GPU_TARGETS}")

###############################################################################
# CREATE ROCSHMEM LIBRARY
###############################################################################
if (NOT BUILD_TESTS_ONLY)
  add_library(${PROJECT_NAME})
  add_library(roc::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

  #############################################################################
  # PACKAGE DEPENDENCIES
  #############################################################################
  # GDA links libnuma directly; hsakmt (USE_SDMA) also depends on numa::numa
  if(USE_GDA OR USE_SDMA)
    find_package(NUMA QUIET HINTS "${ROCM_PATH}/lib/rocm_sysdeps")
    if(NOT TARGET numa::numa)
      find_package(PkgConfig QUIET)
      if(PKG_CONFIG_FOUND)
        pkg_check_modules(_NUMA_PC QUIET numa)
      endif()
      find_library(_NUMA_LIB NAMES numa HINTS ${_NUMA_PC_LIBDIR})
      find_path(_NUMA_INC NAMES numa.h HINTS ${_NUMA_PC_INCLUDEDIR})
      include(FindPackageHandleStandardArgs)
      find_package_handle_standard_args(NUMA DEFAULT_MSG _NUMA_LIB _NUMA_INC)
      if(NUMA_FOUND AND NOT TARGET numa::numa)
        add_library(numa::numa SHARED IMPORTED)
        set_target_properties(numa::numa PROPERTIES
          IMPORTED_LOCATION "${_NUMA_LIB}"
          INTERFACE_INCLUDE_DIRECTORIES "${_NUMA_INC}")
      endif()
    endif()

    if(NOT TARGET numa::numa AND (USE_GDA OR USE_SDMA))
      message(WARNING "libnuma not found. Unable to build GDA or SDMA")
      set(USE_SDMA OFF)
      set(USE_GDA OFF)
      set(GDA_IONIC OFF)
      set(GDA_BNXT OFF)
      set(GDA_MLX5 OFF)
    endif()
  endif()

  if (NOT USE_EXTERNAL_MPI STREQUAL "OFF")
    find_package(MPI)
  else()
    message ("-- External MPI detection disabled by user")
  endif()

  if (MPI_FOUND)
    set(HAVE_EXTERNAL_MPI ON)
  else()
    set(HAVE_EXTERNAL_MPI OFF)
    set(BUILD_UNIT_TESTS OFF)
    set(BUILD_EXAMPLES OFF)
  endif()

  if (USE_EXTERNAL_MPI STREQUAL "ON")
    if(NOT HAVE_EXTERNAL_MPI)
      message(FATAL_ERROR "External MPI support requested but MPI support not found. Build Aborted")
    endif()
  endif()

  if (ASAN)
    message(STATUS "Removing xnack- entries from GPU_TARGETS because they are invalid with AddressSanitizer (ASAN)")
    list(FILTER GPU_TARGETS EXCLUDE REGEX ".*:xnack-$")
    if(GPU_TARGETS_FULL)
      list(FILTER GPU_TARGETS_FULL EXCLUDE REGEX ".*:xnack-$")
    endif()

    # RDNA archs (gfx11xx/gfx12xx) have no xnack+ variant at all, so device
    # ASAN can never be enabled for them - clang just ignores
    # -fsanitize=address for that offload arch with a warning. Drop them
    # instead of leaving a target in GPU_TARGETS that silently gets no
    # instrumentation.
    set(_ASAN_INCOMPATIBLE_ARCHS gfx1100 gfx1201 gfx1250)
    foreach(_asan_incompatible_arch ${_ASAN_INCOMPATIBLE_ARCHS})
      list(FILTER GPU_TARGETS EXCLUDE REGEX "^${_asan_incompatible_arch}(:.*)?$")
      if(GPU_TARGETS_FULL)
        list(FILTER GPU_TARGETS_FULL EXCLUDE REGEX "^${_asan_incompatible_arch}(:.*)?$")
      endif()
    endforeach()

    if(NOT GPU_TARGETS)
      message(FATAL_ERROR "ASAN requires a GPU target that supports xnack+ (e.g. gfx90a, gfx942, gfx950). "
        "None of the requested/default GPU_TARGETS support it.")
    endif()

    # CDNA archs (gfx90a/gfx942/gfx950) only get ASAN instrumentation with the
    # xnack+ feature explicitly enabled; a bare arch name compiles without it.
    # Force :xnack+ onto any entry that doesn't already request it.
    set(_ASAN_GPU_TARGETS "")
    foreach(_asan_target ${GPU_TARGETS})
      if(NOT _asan_target MATCHES "xnack\\+")
        message(STATUS "ASAN: appending :xnack+ to GPU target '${_asan_target}'")
        set(_asan_target "${_asan_target}:xnack+")
      endif()
      list(APPEND _ASAN_GPU_TARGETS "${_asan_target}")
    endforeach()
    set(GPU_TARGETS "${_ASAN_GPU_TARGETS}")

    if(GPU_TARGETS_FULL)
      set(_ASAN_GPU_TARGETS_FULL "")
      foreach(_asan_target ${GPU_TARGETS_FULL})
        if(NOT _asan_target MATCHES "xnack\\+")
          set(_asan_target "${_asan_target}:xnack+")
        endif()
        list(APPEND _ASAN_GPU_TARGETS_FULL "${_asan_target}")
      endforeach()
      set(GPU_TARGETS_FULL "${_ASAN_GPU_TARGETS_FULL}")
    endif()
  endif()

  # hip-config-amd.cmake caches GPU_BUILD_TARGETS without FORCE, so on a
  # reconfigure of an existing build directory it can retain a stale value
  # (e.g. from before ASAN was enabled, or from a previous GPU_TARGETS) even
  # though GPU_TARGETS itself was correctly refreshed above. Force it to match
  # on every configure so hip::device's --offload-arch flags stay in sync.
  set(GPU_BUILD_TARGETS "${GPU_TARGETS}" CACHE STRING "GPU targets to compile for" FORCE)

  find_package(hip REQUIRED)
  find_package(hsa-runtime64 REQUIRED)

  if (USE_SDMA)
    find_package(hsakmt REQUIRED)
    # Workaround for RoCM 7.13/14
    # hsakmtTargets.cmake contains paths baked in from the ROCm RHEL CI build:
    #   /usr/lib64/libc.so      — bare libc path instead of -lc (always implicit)
    #   -L/__w/rockrel/...      — hardcoded CI search path for libdrm/libdrm_amdgpu
    # Strip both and inject the correct search path for libdrm.
    get_target_property(_hsakmt_iface_libs hsakmt::hsakmt INTERFACE_LINK_LIBRARIES)
    list(FILTER _hsakmt_iface_libs EXCLUDE REGEX "(/libc\\.so|/__w/)")
    find_library(_DRM_LIB drm HINTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
    if(_DRM_LIB)
      get_filename_component(_drm_dir "${_DRM_LIB}" DIRECTORY)
      list(PREPEND _hsakmt_iface_libs "-L${_drm_dir}")
      unset(_drm_dir)
    endif()
    set_target_properties(hsakmt::hsakmt PROPERTIES INTERFACE_LINK_LIBRARIES "${_hsakmt_iface_libs}")
    unset(_hsakmt_iface_libs)
  endif()

  if(USE_ROCPROFILER_REGISTER)
    find_package(rocprofiler-register QUIET)
    if(NOT rocprofiler-register_FOUND)
      message(STATUS "rocprofiler-register not found - rocSHMEM tracing disabled.")
    endif()
  endif()

  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)

  # Check if std::filesystem requires linking stdc++fs
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_source_compiles("
    #include <filesystem>
    int main() {
      auto p = std::filesystem::current_path();
      return 0;
    }
  " FILESYSTEM_NO_LINK_NEEDED)
  set(CMAKE_REQUIRED_QUIET FALSE)

  if(${ROCM_MAJOR_VERSION} GREATER_EQUAL 6 )
    set(HAVE_DEVICE_MALLOC_UNCACHED ON)
  elseif (${ROCM_MAJOR_VERSION} EQUAL 5 AND ${ROCM_MINOR_VERSION} GREATER_EQUAL 5)
    set(HAVE_DEVICE_MALLOC_UNCACHED ON)
  else()
    set(HAVE_DEVICE_MALLOC_UNCACHED OFF)
  endif()

  execute_process(
    COMMAND git -C "${PROJECT_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE ROCSHMEM_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT ROCSHMEM_GIT_HASH)
    set(ROCSHMEM_GIT_HASH "unknown")
  endif()

  string(REPLACE ";" " " ROCSHMEM_OFFLOAD_TARGETS "${GPU_TARGETS}")

  set(_backends "")
  if(USE_GDA)
    set(_gda_nics "")
    foreach(_nic IN ITEMS IONIC BNXT MLX5)
      if(GDA_${_nic})
        list(APPEND _gda_nics "${_nic}")
      endif()
    endforeach()
    list(JOIN _gda_nics "/" _gda_providers)
    list(APPEND _backends "GDA(${_gda_providers})")
  endif()
  if(USE_RO)
    list(APPEND _backends "RO")
  endif()
  if(USE_IPC)
    list(APPEND _backends "IPC")
  endif()

  if(_backends)
    list(JOIN _backends "+" ROCSHMEM_BACKEND)
  else()
    set(ROCSHMEM_BACKEND "NONE")
  endif()
  set(ROCSHMEM_VENDOR_STRING "rocSHMEM_v${VERSION_STRING}_${ROCSHMEM_BACKEND}")

  configure_file(cmake/rocshmem_config.h.in include/rocshmem/rocshmem_config.h)

  # Embed rocshmem_config.h as a C++17 raw string literal so build_info.cpp
  # (and the rocshmem_info tool) can display it without reading the file at runtime.
  file(READ "${CMAKE_BINARY_DIR}/include/rocshmem/rocshmem_config.h" _config_content)
  file(WRITE "${CMAKE_BINARY_DIR}/include/rocshmem/rocshmem_config_embedded.hpp"
    "// Auto-generated from rocshmem_config.h — do not edit.\n"
    "#pragma once\n"
    "static const char rocshmem_config_h_content[] = R\"ROCSHMEM_CFG_END(\n"
    "${_config_content}"
    ")ROCSHMEM_CFG_END\";\n"
  )

  #############################################################################
  # LINKING AND INCLUDE DIRECTORIES
  #############################################################################
  target_compile_options(
    ${PROJECT_NAME}
    PUBLIC
      -fgpu-rdc
      $<$<BOOL:${ASAN}>:-fsanitize=address -g>
    PRIVATE
      -Wall
      -Wextra
      -fmacro-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=
  )

  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>            # rocshmem_config.h
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/rocshmem>   # rocshmem_config.h from rocshmem.hpp
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
      $<$<BOOL:${HAVE_EXTERNAL_MPI}>:MPI::MPI_CXX>
      Threads::Threads
      hip::device
      hip::host
      dl
      hsa-runtime64::hsa-runtime64
      -fgpu-rdc
      $<$<NOT:$<BOOL:${FILESYSTEM_NO_LINK_NEEDED}>>:stdc++fs>
      $<$<BOOL:${USE_SDMA}>:hsakmt::hsakmt>
    PRIVATE
      $<$<OR:$<BOOL:${USE_GDA}>,$<BOOL:${USE_SDMA}>>:numa::numa>
  )

  target_link_options(
    ${PROJECT_NAME}
    PUBLIC
      $<$<BOOL:${ASAN}>:-fsanitize=address -g -shared-libsan>
  )

  if (ASAN)
    # Add ASAN runtime directory to RPATH so executables can find libclang_rt.asan at runtime
    target_link_options(
      ${PROJECT_NAME}
      PUBLIC
        -Wl,-rpath,${ASAN_RUNTIME_DIR}
    )
  endif()

  if(USE_ROCPROFILER_REGISTER AND TARGET rocprofiler-register::rocprofiler-register)
    target_link_libraries(${PROJECT_NAME}
                          PRIVATE rocprofiler-register::rocprofiler-register)
  endif()

  if(${ROCM_MAJOR_VERSION} LESS 7)
    # ROCm 6.x requires us to explicitly enable warp sync builtins
    target_compile_definitions(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:HIP_ENABLE_WARP_SYNC_BUILTINS=1>)
  endif()

  #############################################################################
  # DEVICE BITCODE COMPILATION
  #############################################################################
  include(${CMAKE_SOURCE_DIR}/cmake/DeviceBitcode.cmake)

  #############################################################################
  # INSTALL
  #############################################################################
  include(ROCMInstallTargets)
  include(ROCMCreatePackage)

  rocm_install(TARGETS rocshmem)

  rocm_install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  rocm_install(
    FILES "${CMAKE_BINARY_DIR}/include/rocshmem/rocshmem_config.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocshmem
  )

  if (BUILD_TOOLS)
    add_subdirectory(tools)
    rocm_install(
      PROGRAMS "${CMAKE_BINARY_DIR}/tools/rocshmem_info"
      DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  endif()

  rocm_package_add_dependencies(
    DEPENDS
      hsa-rocr
      hip-runtime-amd
      rocm-dev
  )

  set(_rocshmem_export_deps)
  if(USE_SDMA AND TARGET hsakmt::hsakmt)
    list(APPEND _rocshmem_export_deps PACKAGE hsakmt)
  endif()
  if(USE_ROCPROFILER_REGISTER AND TARGET rocprofiler-register::rocprofiler-register)
    list(APPEND _rocshmem_export_deps PACKAGE rocprofiler-register)
  endif()

  # ROCm 7.13+ ships NUMAConfig.cmake in rocm_sysdeps so find_dependency(NUMA)
  # works cleanly for consumers. On older ROCm, bake the resolved paths into an
  # included file since no config-mode NUMA package exists.
  # For ROCm >= 7.13 NUMAConfig.cmake ships in rocm_sysdeps alongside the ROCm
  # install. Inject a CMAKE_PREFIX_PATH prepend into the generated config using
  # the hip package location (stable relative to rocm_sysdeps) so internal
  # find_dependency(NUMA) resolves without requiring downstream project awareness.
  # For older ROCm, bake the resolved numa::numa target directly.
  set(_rocshmem_numa_include)
  if(${ROCM_MAJOR_VERSION} GREATER 7 OR
     (${ROCM_MAJOR_VERSION} EQUAL 7 AND ${ROCM_MINOR_VERSION} GREATER_EQUAL 13))
    if(USE_GDA OR USE_SDMA)
      list(APPEND _rocshmem_export_deps PACKAGE NUMA)
    endif()
  elseif(TARGET numa::numa)
    # Older ROCm: bake resolved paths so consumers need no FindNUMA.cmake.
    get_target_property(_numa_lib numa::numa IMPORTED_LOCATION)
    get_target_property(_numa_inc numa::numa INTERFACE_INCLUDE_DIRECTORIES)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rocshmem-numa-targets.cmake"
      "if(NOT TARGET numa::numa)\n"
      "  add_library(numa::numa SHARED IMPORTED)\n"
      "  set_target_properties(numa::numa PROPERTIES\n"
      "    IMPORTED_LOCATION \"${_numa_lib}\"\n"
      "    INTERFACE_INCLUDE_DIRECTORIES \"${_numa_inc}\")\n"
      "endif()\n"
      "set(NUMA_FOUND TRUE)\n")
    unset(_numa_lib)
    unset(_numa_inc)
    set(_rocshmem_numa_include "${CMAKE_CURRENT_BINARY_DIR}/rocshmem-numa-targets.cmake")
  endif()

  set(_rocshmem_include_arg)
  if(_rocshmem_numa_include)
    set(_rocshmem_include_arg INCLUDE "${_rocshmem_numa_include}")
  endif()
  rocm_export_targets(
    TARGETS roc::rocshmem
    NAMESPACE roc::
    DEPENDS ${_rocshmem_export_deps}
    ${_rocshmem_include_arg}
  )
  unset(_rocshmem_include_arg)
  unset(_rocshmem_export_deps)
  # Patch the generated config to expand CMAKE_PREFIX_PATH with rocm_sysdeps
  # before find_dependency(NUMA) so NUMAConfig.cmake is found on ROCm >= 7.13
  # without consumers needing to add rocm_sysdeps to their CMAKE_PREFIX_PATH.
  if(${ROCM_MAJOR_VERSION} GREATER 7 OR
     (${ROCM_MAJOR_VERSION} EQUAL 7 AND ${ROCM_MINOR_VERSION} GREATER_EQUAL 13))
    set(_config_out "${CMAKE_CURRENT_BINARY_DIR}/rocshmem-config.cmake")
    file(READ "${_config_out}" _config_content)
    string(REPLACE
      "find_dependency(NUMA)"
      "string(REGEX REPLACE \"/lib/cmake/hip$\" \"\" _rocshmem_rocm_root \"\${hip_DIR}\")\nlist(APPEND CMAKE_PREFIX_PATH \"\${_rocshmem_rocm_root}/lib/rocm_sysdeps\")\nunset(_rocshmem_rocm_root)\nfind_dependency(NUMA)\nlist(REMOVE_AT CMAKE_PREFIX_PATH -1)"
      _config_content "${_config_content}")
    file(WRITE "${_config_out}" "${_config_content}")
    unset(_config_out)
    unset(_config_content)
  endif()
  include(ROCMPackageConfigHelpers)
  include(ROCMClients)
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(tests PACKAGE_NAME tests)

  rocm_create_package(
    NAME "rocSHMEM"
    DESCRIPTION "ROCm OpenSHMEM (rocSHMEM)"
    MAINTAINER "rocSHMEM Maintainer <rocshmem-maintainer@amd.com>"
  )
endif (NOT BUILD_TESTS_ONLY)

###############################################################################
# TEST SUBDIRECTORIES
###############################################################################
if (NOT BUILD_TESTS_ONLY)
  add_subdirectory(src)
endif (NOT BUILD_TESTS_ONLY)

###############################################################################
# TEST SUBDIRECTORIES
###############################################################################
if(BUILD_CTESTS)
    enable_testing()
endif()
add_subdirectory(tests)

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# NOTE: The rocshmem4py Python binding is no longer built from this project.
# It now lives as a standalone Python project at rocm-systems/python/rocshmem
# and builds independently against an installed rocSHMEM via
# `find_package(rocshmem CONFIG)` (CMAKE_PREFIX_PATH). See that project's
# README for build/packaging.

