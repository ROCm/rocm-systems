# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.

# CMake version minimum requirements
#==================================================================================================
cmake_minimum_required(VERSION 3.16)

# CMake Toolchain file to define compilers and path to ROCm
#==================================================================================================
if (NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/toolchain-linux.cmake")
  message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# RCCL project
#==================================================================================================
project(rccl CXX)

# Build options
#==================================================================================================
option(BUILD_ADDRESS_SANITIZER                 "Enable address sanitizer"                      OFF)
option(BUILD_BFD                               "Enable custom backtrace (if bfd.h exists)"     OFF)
option(BUILD_LOCAL_GPU_TARGET_ONLY             "Build only for GPUs detected on this machine"  OFF)
option(BUILD_SHARED_LIBS                       "Build as shared library"                       ON)
option(BUILD_TESTS                             "Build unit test programs"                      OFF)
option(COLLTRACE                               "Collective Trace Option"                       ON)
option(DUMP_ASM                                "Disassemble and dump"                          OFF)
option(ENABLE_CODE_COVERAGE                    "Enable code coverage"                          OFF)
option(ENABLE_MSCCL_KERNEL                     "Enable MSCCL while compiling"                  OFF)
option(ENABLE_MSCCLPP                          "Enable MSCCL++"                                OFF)
option(ENABLE_MSCCLPP_CLIP                     "Enable MSCCL++ CLIP"                           OFF)
option(ENABLE_MSCCLPP_EXECUTOR                 "Enable MSCCL++ Executor"                       OFF)
option(ENABLE_MSCCLPP_FORMAT_CHECKS            "Enable formatting checks in MSCCL++"           OFF)
option(MSCCLPP_APPLY_PATCHES                   "Apply source code patches to MSCCL++"          ON)
option(ENABLE_NPKIT                            "Enable NPKit"                                  OFF)
option(ENABLE_IFC                              "Enable indirect function call"                 OFF)
option(GENERATE_SYM_KERNELS                    "Generate symmetric memory kernels"             OFF)
option(INSTALL_DEPENDENCIES                    "Force install dependencies"                    OFF)
option(REPORT_KERNEL_RESOURCE_USE              "Append -Rpass-analysis=kernel to CXX flags"    OFF)
option(ROCTX                                   "Enable ROCTX"                                  ON)
option(PROFILE                                 "Enable profiling"                              OFF)
option(TIMETRACE                               "Enable time-trace during compilation"          OFF)
option(TRACE                                   "Enable additional tracing"                     OFF)
option(FAULT_INJECTION                         "Enable fault injection"                        ON)
option(QUIET_WARNINGS                          "Supress compiler warnings"                     OFF)
option(ENABLE_ROCSHMEM                         "Enable rocSHMEM support in RCCL"               OFF)

# Default GPU architectures to build
#==================================================================================================
set(DEFAULT_GPUS
      gfx906
      gfx908
      gfx90a
      gfx942
      gfx950
      gfx1030
      gfx1100
      gfx1101
      gfx1102
      gfx1200
      gfx1201)

# Load CMake modules
#==================================================================================================
include(CheckIncludeFiles)
include(CheckSymbolExists)
include(cmake/Dependencies.cmake) # GTest, rocm-cmake, rocm_local_targets
include(cmake/CheckSymbolExistsNoWarn.cmake)
include(cmake/MSCCLPP.cmake)

# Include rocSHMEM build module only if enabled
if(ENABLE_ROCSHMEM)
  include(cmake/ROCSHMEM.cmake)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Build only for local GPU architecture
if (BUILD_LOCAL_GPU_TARGET_ONLY)
  message(STATUS "Building only for local GPU target")
  if (COMMAND rocm_local_targets)
    rocm_local_targets(DEFAULT_GPUS)
  else()
    message(WARNING "Unable to determine local GPU targets. Falling back to default GPUs.")
  endif()
endif()

# Determine which GPU architectures to build for
set(GPU_TARGETS "${DEFAULT_GPUS}" CACHE STRING "Target default GPUs if GPU_TARGETS is not defined.")

# ROCM NetIB patch
include(cmake/rocmIb.cmake)

# Modify GPU architectures for Address Sanitizer builds by appending "xnack+"
if (BUILD_ADDRESS_SANITIZER)
  SET(amdgpu_targets "")
  foreach(amdgpu_target IN LISTS GPU_TARGETS)
    if(NOT amdgpu_target STREQUAL "")
      string(FIND "${amdgpu_target}" ":xnack+" HAS_XNACK_SUFFIX)
      if(HAS_XNACK_SUFFIX EQUAL -1)
        list(APPEND amdgpu_targets "${amdgpu_target}:xnack+")
      else()
        list(APPEND amdgpu_targets "${amdgpu_target}")
      endif()
    endif()
  endforeach()
  SET(GPU_TARGETS "${amdgpu_targets}")
endif()

# Check if clang compiler can offload to GPU_TARGETS
if (COMMAND rocm_check_target_ids)
  message(STATUS "Checking for ROCm support for GPU targets: " "${GPU_TARGETS}")
  rocm_check_target_ids(SUPPORTED_GPUS TARGETS ${GPU_TARGETS})
else()
  message(WARNING "Unable to check for supported GPU targets. Falling back to default GPUs.")
  set(SUPPORTED_GPUS ${DEFAULT_GPUS})
endif()

set(GPU_TARGETS "${SUPPORTED_GPUS}")
message(STATUS "Compiling for ${GPU_TARGETS}")

## NOTE: Reload rocm-cmake in order to update GPU_TARGETS
include(cmake/Dependencies.cmake) # Reloading to use desired GPU_TARGETS instead of defaults

# Try to establish ROCM_PATH (for find_package)
#==================================================================================================
if(NOT DEFINED ROCM_PATH)
  # Guess default location
  set(ROCM_PATH "/opt/rocm")
  message(WARNING "Unable to find ROCM_PATH: Falling back to ${ROCM_PATH}")
else()
  message(STATUS "ROCM_PATH found: ${ROCM_PATH}")
endif()
set(ENV{ROCM_PATH} ${ROCM_PATH})

if("${CMAKE_CXX_COMPILER}" MATCHES ".*amdclang\\+\\+")
  message(STATUS "Compiling with amdclang++")
  set(COMPILER_EXE_NAME amdclang++)
  set(COMPILER_GREP_STRING "AMD clang version")
  set(COMPILER_AWK_CMD "awk -F\" \" '{ printf $4}'")
elseif("${CMAKE_CXX_COMPILER}" MATCHES ".*clang\\+\\+")
  message(STATUS "Compiling with clang++")
  set(COMPILER_EXE_NAME clang++)
  set(COMPILER_GREP_STRING "AMD clang version")
  set(COMPILER_AWK_CMD "awk -F\" \" '{ printf $4}'")
elseif("${CMAKE_CXX_COMPILER}" MATCHES ".*hipcc$")
  message(STATUS "Compiling with hipcc")
  set(COMPILER_EXE_NAME hipcc)
  set(COMPILER_GREP_STRING "HIP version")
  set(COMPILER_AWK_CMD "awk -F\" \" '{ printf $3}' | awk -F\"-\" '{ printf $1}'")
else()
  message(FATAL_ERROR "RCCL can be built only with hipcc or amdclang++")
endif()

# Set CMAKE flags
#==================================================================================================
set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "")
set(CMAKE_CXX_STANDARD   17)   # We use C++17 features, this will add compile option: -std=c++17
set(CMAKE_CXX_EXTENSIONS OFF)  # Without this line, it will add -std=gnu++17 instead, which has some issues.
if(ROCM_PATH)
  list(APPEND CMAKE_PREFIX_PATH  # Add ROCM_PATH to CMake search paths (for finding HIP / HSA
              ${ROCM_PATH}
              ${ROCM_PATH}/hip
              ${ROCM_PATH}/llvm)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3" CACHE STRING "Release flags" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g" CACHE STRING "RelWithDebInfo flags" FORCE)
set(CMAKE_C_FLAGS_RELEASE "-O3" CACHE STRING "Release flags" FORCE)
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g" CACHE STRING "RelWithDebInfo flags" FORCE)

# Check for required dependencies
#==================================================================================================
## Check for Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

## Check for HIP
find_package(hip REQUIRED)
message(STATUS "HIP compiler:     ${HIP_COMPILER}")
message(STATUS "HIP runtime:      ${HIP_RUNTIME}")
if(NOT "${HIP_COMPILER}" MATCHES "clang")
  message(FATAL_ERROR "RCCL requires clang-based compiler (amdclang++ or hipcc)")
endif()

## Check for compiler version
find_program(compiler_executable ${COMPILER_EXE_NAME})
message(STATUS "${COMPILER_EXE_NAME} executable: ${compiler_executable}")
execute_process(
  COMMAND         bash "-c" "${compiler_executable} --version | grep \"${COMPILER_GREP_STRING}\" | ${COMPILER_AWK_CMD}"
  OUTPUT_VARIABLE compiler_version_string)
message(STATUS "${COMPILER_EXE_NAME} version:    ${compiler_version_string}")

## Check for HIP version
find_program(hipconfig_executable hipconfig)
message(STATUS "hipconfig executable: ${hipconfig_executable}")
execute_process(
  COMMAND         bash "-c" "${hipconfig_executable} -v | awk -F\"-\" '{ printf $1 }'"
  OUTPUT_VARIABLE hip_version_string)
message(STATUS "${COMPILER_EXE_NAME} HIP version:    ${hip_version_string}")

## Check for ROCm version
set(EXPLICIT_ROCM_VERSION "" CACHE STRING "Explicit ROCM version to compile to (auto detect if empty)")
if(NOT DEFINED ROCMCORE_PATH)
  set(ROCMCORE_PATH "${ROCM_PATH}" CACHE PATH "Path to ROCm core")
endif()

if(EXPLICIT_ROCM_VERSION)
  set(rocm_version_string "${EXPLICIT_ROCM_VERSION}")
elseif(ROCMCORE_PATH)
  message(STATUS "Reading ROCM version from ${ROCMCORE_PATH}/.info/version")
  file(READ "${ROCMCORE_PATH}/.info/version" rocm_version_string)
else()
  message(FATAL_ERROR "Could not determine ROCM version (set EXPLICIT_ROCM_VERSION or set ROCM_PATH to a valid installation)")
endif()
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" rocm_version_matches ${rocm_version_string})
if (rocm_version_matches)
    set(ROCM_MAJOR_VERSION ${CMAKE_MATCH_1})
    set(ROCM_MINOR_VERSION ${CMAKE_MATCH_2})
    set(ROCM_PATCH_VERSION ${CMAKE_MATCH_3})

    message(STATUS "ROCm version: ${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}")

    # Convert the version components to int for comparison
    math(EXPR ROCM_VERSION "(10000 * ${ROCM_MAJOR_VERSION}) + (100 * ${ROCM_MINOR_VERSION}) + ${ROCM_PATCH_VERSION}")
    add_definitions("-DROCM_VERSION=${ROCM_VERSION}")
else()
    message(WARNING "Failed to extract ROCm version.")
endif()

### Required for checking HIP device symbols when building with amdclang++
set(CMAKE_REQUIRED_LIBRARIES hip::device)

### Check for hipDeviceMallocUncached support
check_symbol_exists("hipDeviceMallocUncached" "hip/hip_runtime_api.h" HIP_UNCACHED_MEMORY)

### Check for hipHostMallocUncached support
check_symbol_exists("hipHostMallocUncached" "hip/hip_runtime_api.h" HIP_HOST_UNCACHED_MEMORY)

### Check for hipDeviceMallocContiguous support
check_symbol_exists("hipDeviceMallocContiguous" "hip/hip_runtime_api.h" HIP_CONTIGUOUS_MEMORY)

unset(CMAKE_REQUIRED_LIBRARIES)

### Check for indirect function call support
if(ENABLE_IFC)
  if("${hip_version_string}" VERSION_GREATER_EQUAL "5.5.30201")
    set(IFC_ENABLED ON)
    message(STATUS "Indirect function call enabled")
  else()
    set(IFC_ENABLED OFF)
    message(WARNING "Indirect function call disabled - requires HIP version >= 5.5.30201")
  endif()
else()
  set(IFC_ENABLED OFF)
endif()

## Check for LL128 support
if("${hip_version_string}" VERSION_GREATER_EQUAL "6.1.33591")
  set(LL128_ENABLED ON)
  message(STATUS "RCCL LL128 protocol enabled")
else()
  message(STATUS "RCCL LL128 protocol disabled - requires HIP version >= 6.1.33591")
endif()

## Check for hsa-runtime64
find_package(hsa-runtime64 REQUIRED)
get_target_property(HSA_INCLUDE_PATH hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "HSA runtime: ${HSA_INCLUDE_PATH}")

## Check for amd-smi if ROCm 7.11.0 or newer
if(ROCM_VERSION VERSION_GREATER_EQUAL "71100")
  find_package(amd_smi PATHS ${ROCM_PATH}/lib/cmake/amd_smi)
  if(amd_smi_FOUND)
    message(STATUS "amd_smi_INCLUDE_DIR: ${amd_smi_INCLUDE_DIR}")
    message(STATUS "amd_smi_LIB_DIR: ${amd_smi_LIB_DIR}")
    set(SMI_INCLUDE_DIR "${amd_smi_INCLUDE_DIR}" CACHE INTERNAL "amd-smi include directory")
    set(SMI_LIB_DIR "${amd_smi_LIB_DIR}" CACHE INTERNAL "amd-smi library directory")
    set(SMI_LIB_NAME "amd-smi-lib" CACHE INTERNAL "amd-smi-lib for packaging")
    if(NOT EXISTS "${SMI_INCLUDE_DIR}" OR NOT EXISTS "${SMI_LIB_DIR}")
      message(FATAL_ERROR "amd_smi not found in ${SMI_INCLUDE_DIR}")
    endif()
    message(STATUS "Found amd_smi at ${SMI_INCLUDE_DIR}")
    set(SMI_LIBRARIES amd_smi)
    set(USE_AMDSMI ON CACHE INTERNAL "Use amd-smi instead of rocm-smi")
  endif()
endif()

if(NOT USE_AMDSMI)
  ## Fallback to rocm-smi if amd-smi not found or ROCm < 7.11.0
  message(WARNING "Could not find amd_smi. Falling back to rocm_smi.")
  find_package(rocm_smi PATHS ${ROCM_PATH}/lib/cmake/rocm_smi)
  if(rocm_smi_FOUND)
    set(SMI_INCLUDE_DIR "${rocm_smi_INCLUDE_DIR}" CACHE INTERNAL "rocm-smi include directory")
    set(SMI_LIB_DIR "${rocm_smi_LIB_DIR}" CACHE INTERNAL "rocm-smi library directory")
  else()
    message(WARNING "CMake could not find rocm-smi. Checking old include directory structure for rocm_smi")
    set(SMI_INCLUDE_DIR "${ROCM_PATH}/rocm_smi/include")
    set(SMI_LIB_DIR     "${ROCM_PATH}/rocm_smi/lib")
  endif()

  if(NOT EXISTS "${SMI_INCLUDE_DIR}" OR NOT EXISTS "${SMI_LIB_DIR}")
    message(FATAL_ERROR "rocm_smi not found in ${SMI_INCLUDE_DIR}")
  endif()
  message(STATUS "Found rocm_smi at ${SMI_INCLUDE_DIR}")
  set(SMI_LIB_NAME "rocm-smi-lib" CACHE INTERNAL "rocm-smi-lib for packaging")
  set(SMI_LIBRARIES rocm_smi64)

  check_include_file_cxx("${SMI_INCLUDE_DIR}/rocm_smi/rocm_smi64Config.h" HAVE_ROCM_SMI64CONFIG)

  ### Check for RSMI_INIT_FLAG_THRAD_ONLY_MUTEX support
  file(READ "${SMI_INCLUDE_DIR}/rocm_smi/rocm_smi.h" rocm_smi_incl)
  string(FIND "${rocm_smi_incl}" "RSMI_INIT_FLAG_THRAD_ONLY_MUTEX" matchres)
  if(${matchres} EQUAL -1)
    message(STATUS "RSMI_INIT_FLAG_THRAD_ONLY_MUTEX not supported")
  else()
    message(STATUS "RSMI_INIT_FLAG_THRAD_ONLY_MUTEX supported")
    set(HAVE_ROCM_SMI_THREAD_ONLY_MUTEX True)
  endif ()
endif()

## Check for BFD library if custom backtrace is requested
if(BUILD_BFD)
  enable_language(C)
  check_include_files(bfd.h HAVE_BFD)
  if (HAVE_BFD)
    message(STATUS "-- Found BFD support")

    ### Required for checking HIP device symbols when building with amdclang++
    set(CMAKE_REQUIRED_LIBRARIES hip::device)

    # Check for specific BFD feature support
    CHECK_SYMBOL_EXISTS(bfd_get_section_flags "bfd.h" HAVE_DECL_BFD_GET_SECTION_FLAGS)
    CHECK_SYMBOL_EXISTS(bfd_get_section_vma   "bfd.h" HAVE_DECL_BFD_GET_SECTION_VMA)
    CHECK_CXX_SOURCE_COMPILES(
      "#include <bfd.h>

       int main (int argc, char **argv){
           bfd_size_type size;
           bfd abfd;
           asection sec;
           size = bfd_section_size(&abfd, &sec);
           return (int)(size);
       }"
      HAVE_TWO_ARG_BFD_SECTION_SIZE)

    unset(CMAKE_REQUIRED_LIBRARIES)

    # Check for iberty support
    find_library(HAVE_IBERTY iberty PATHS /usr/lib64 /usr/lib/ PATH_SUFFIXES x86_64-linux-gnu)
    if(HAVE_IBERTY)
      message(STATUS "iberty found @ ${HAVE_IBERTY}")
    endif()

    # Check for demangle support
    find_path(DEMANGLE_DIR demangle.h PATHS /usr/include PATH_SUFFIXES libiberty)
    if(NOT DEMANGLE_DIR)
      message(WARNING "Could not find demangle.h ${DEMANGLE_DIR}")
    else()
      message(STATUS "Found demangle.h in ${DEMANGLE_DIR}")
    endif()
  else()
    message(WARNING "bfd.h header not found - Disabling custom backtrace")
  endif()
endif()



# Check for --amdgpu-kernarg-preload-count
check_cxx_compiler_flag("-mllvm --amdgpu-kernarg-preload-count=16" HAVE_KERNARG_PRELOAD)
if (HAVE_KERNARG_PRELOAD)
  message(STATUS "Kernarg preloading to SGPR enabled")
endif()

check_cxx_compiler_flag("-parallel-jobs=12" HAVE_PARALLEL_JOBS)
if (HAVE_PARALLEL_JOBS)
  message(STATUS "Parallel jobs enabled")
endif()

## Disable building MSCCL++ if the build environment is invalid
## Currently MSCCL++ is supported only on gfx942 and gfx950, and only on Ubuntu and CentOS
set(MSCCLPP_SUPPORTED_ARCHS "gfx942" "gfx942:xnack-" "gfx942:xnack+" "gfx950" "gfx950:xnack-" "gfx950:xnack+")


# Check if any of the supported architectures are in GPU_TARGETS
set(ARCH_MATCH_FOUND OFF)
set(MSCCLPP_GPU_TARGETS "")
foreach(ARCH IN LISTS GPU_TARGETS)
  if(ARCH IN_LIST MSCCLPP_SUPPORTED_ARCHS)
    set(ARCH_MATCH_FOUND ON)
    list(APPEND MSCCLPP_GPU_TARGETS "${ARCH}")
  endif()
endforeach()
set(MSCCLPP_GPU_TARGETS "${MSCCLPP_GPU_TARGETS}" CACHE STRING "GPU Targets supported by MSCCL++" FORCE)

if (ENABLE_MSCCLPP AND NOT ARCH_MATCH_FOUND)
  set(ENABLE_MSCCLPP OFF)
  message(WARNING "Can only build MSCCL++ for supported GPU_TARGETS: ${MSCCLPP_SUPPORTED_ARCHS}; current GPU_TARGETS: ${GPU_TARGETS}; so disabling MSCCL++ build")
endif()

# MSCCL++ is only supported on ROCm 6.2.0 or newer
if (ENABLE_MSCCLPP AND ROCM_VERSION VERSION_LESS "60200")
  set(ENABLE_MSCCLPP OFF)
  message(WARNING "MSCCL++ integration only supported on ROCm 6.2.0 or greater; disabling MSCCL++ build")
endif()

## Disable WARP_SPEED if the build environment is invalid
set(WARP_SPEED_SUPPORTED_ARCHS "gfx950" "gfx950:xnack-" "gfx950:xnack+")
set(ARCH_MATCH_FOUND OFF)
foreach(ARCH IN LISTS GPU_TARGETS)
  if(ARCH IN_LIST WARP_SPEED_SUPPORTED_ARCHS)
    set(ARCH_MATCH_FOUND ON)
  endif()
endforeach()
if (NOT ARCH_MATCH_FOUND)
  set(ENABLE_WARP_SPEED OFF)
  message(WARNING "Can only build WARP_SPEED for supported GPU_TARGETS: ${WARP_SPEED_SUPPORTED_ARCHS}; current GPU_TARGETS: ${GPU_TARGETS}; so disabling WARP_SPEED build")
endif()


# cmake_host_system_information(RESULT HOST_OS_ID QUERY DISTRIB_ID) ## Requires cmake 3.22
execute_process(
  COMMAND bash -c "grep '^ID=' /etc/os-release | cut -d'=' -f2 | cut -d'\"' -f2"
  OUTPUT_VARIABLE HOST_OS_ID
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND bash -c "grep '^ID_LIKE=' /etc/os-release | cut -d'=' -f2 | cut -d'\"' -f2"
  OUTPUT_VARIABLE HOST_OS_FAMILY
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (ENABLE_MSCCLPP AND NOT(${HOST_OS_ID} STREQUAL "ubuntu" OR ${HOST_OS_ID} STREQUAL "centos"))
  set(ENABLE_MSCCLPP OFF)
  message(WARNING "MSCCL++ integration not supported on this OS (${HOST_OS_ID}); disabling MSCCL++ build")
endif()

# Check for ROCTX
if(ROCTX)
  find_library(ROCTX_LIB NAMES roctx64)
  find_path(ROCTRACER_INCLUDE_DIR "roctracer/roctx.h")
  if(ROCTX_LIB AND ROCTRACER_INCLUDE_DIR)
    set(ROCTX_ENABLE ON)
    message(STATUS "ROCTX include directory found: ${ROCTRACER_INCLUDE_DIR}")
    message(STATUS "ROCTX library found: ${ROCTX_LIB}")
  else()
    message(WARNING "ROCTX library not found. Skipping ROCTX linking.")
  endif()
endif()

# Determine version from makefiles/version.mk and fill in templates
#==================================================================================================
## parse version from Makefile NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH must exist
## NCCL_SUFFIX is optional
## NCCL_VERSION formatting is ((X) * 1000 + (Y) * 100 + (Z)) so we must first detect one or two digits first
file(READ makefiles/version.mk version_mk_text)
if("${version_mk_text}" MATCHES "NCCL_MAJOR *:= *([0-9]*)")
  set(NCCL_MAJOR ${CMAKE_MATCH_1})
else()
  message(FATAL_ERROR "Failed to parse NCCL_MAJOR")
endif()
if("${version_mk_text}" MATCHES "NCCL_MINOR *:= *([0-9]*)")
  set(NCCL_MINOR ${CMAKE_MATCH_1})
else()
  message(FATAL_ERROR "Failed to parse NCCL_MINOR")
endif()
if("${version_mk_text}" MATCHES "NCCL_PATCH *:= *([0-9]*)")
  set(NCCL_PATCH ${CMAKE_MATCH_1})
else()
  message(FATAL_ERROR "Failed to parse NCCL_PATCH")
endif()
if("${version_mk_text}" MATCHES "NCCL_SUFFIX *:= *([0-9]*)")
  set(NCCL_SUFFIX ${CMAKE_MATCH_1})
else()
  set(NCCL_SUFFIX)
endif()
if("${version_mk_text}" MATCHES "PKG_REVISION *:= *([0-9]*)")
  set(PKG_REVISION ${CMAKE_MATCH_1})
else()
  message(FATAL_ERROR "Failed to parse PKG_REVISION")
endif()
if("${NCCL_PATCH}" MATCHES "[0-9][0-9]")
  set(NCCL_VERSION "${NCCL_MAJOR}${NCCL_MINOR}${NCCL_PATCH}")
else()
  set(NCCL_VERSION "${NCCL_MAJOR}${NCCL_MINOR}0${NCCL_PATCH}")
endif()

## Setup VERSION
set(VERSION_STRING "${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}")
rocm_setup_version(VERSION ${VERSION_STRING})

## Fill in version information for main header file
configure_file(src/nccl.h.in ${PROJECT_BINARY_DIR}/include/rccl/rccl.h) # For external linking
configure_file(src/nccl.h.in ${PROJECT_BINARY_DIR}/include/nccl.h)      # Used by some internal files

# Collect list of all source files
#==================================================================================================
# E.g: find src -type f \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) | sort
set(SRC_FILES
  src/allocator.cc
  src/bootstrap.cc
  src/ce_coll.cc
  src/channel.cc
  src/collectives.cc
  src/commDump.cc
  src/debug.cc
  src/dev_runtime.cc
  src/enqueue.cc
  src/group.cc
  src/init.cc
  src/init_nvtx.cc
  src/mnnvl.cc
  src/msccl.cc
  src/proxy.cc
  src/rccl_wrap.cc
  src/sym_kernels.cc
  src/transport.cc
  src/device/all_gather.h
  src/device/all_reduce.h
  src/device/alltoall_pivot.h
  src/device/alltoall_gda.h
  src/device/broadcast.h
  src/device/common.h
  src/device/common_kernel.h
  src/device/op128.h
  src/device/primitives.h
  src/device/prims_ll128.h
  src/device/prims_ll.h
  src/device/prims_simple.h
  src/device/reduce.h
  src/device/reduce_kernel.h
  src/device/reduce_scatter.h
  src/device/rccl_metadata.h
  src/device/rccl_ptr.h
  src/device/sendrecv.h
  src/device/common.cu
  src/device/onerank.cu
  src/device/network/unpack/unpack_defs.h
  src/device/network/unpack/unpack.h
  src/device/symmetric/all_gather.cuh
  src/device/symmetric/all_reduce.cuh
  src/device/symmetric/kernel.cuh
  src/device/symmetric/primitives.cuh
  src/device/symmetric/reduce_scatter.cuh
  src/graph/connect.cc
  src/graph/paths.cc
  src/graph/rings.cc
  src/graph/rings.h
  src/graph/rome_models.cc
  src/graph/rome_models.h
  src/graph/search.cc
  src/graph/topo.cc
  src/graph/topo.h
  src/graph/trees.cc
  src/graph/tuning.cc
  src/graph/xml.cc
  src/graph/xml.h
  src/include/alloc.h
  src/include/allocator.h
  src/include/alt_rsmi.h
  src/include/archinfo.h
  src/include/api_trace.h
  src/include/argcheck.h
  src/include/BfdBacktrace.hpp
  src/include/bitops.h
  src/include/bootstrap.h
  src/include/ce_coll.h
  src/include/channel.h
  src/include/checks.h
  src/include/collectives.h
  src/include/coll_net.h
  src/include/comm.h
  src/include/core.h
  src/include/cpuset.h
# src/include/cudawrap.h
  src/include/debug.h
  src/include/dev_runtime.h
  src/include/device.h
  src/include/enqueue.h
  src/include/gdrwrap.h
  src/include/git_version.h
  src/include/graph.h
  src/include/group.h
  src/include/hip_rocm_version_info.h
  src/include/ibvcore.h
  src/include/ibvsymbols.h
  src/include/ibvwrap.h
  src/include/info.h
  src/include/ipcsocket.h
  src/include/mnnvl.h
  src/include/nccl_common.h
  src/include/nccl_device.h
  src/include/net_device.h
  src/include/net.h
  src/include/nvmlwrap.h
  src/include/nvtx.h
  src/include/nvtx_payload_schemas.h
  src/include/nvtx_stub.h
  src/include/p2p.h
  src/include/param.h
  src/include/profiler.h
  src/include/proxy.h
  src/include/ras.h
  src/include/rccl_common.h
  src/include/rccl_vars.h
  src/include/register.h
  src/include/register_inline.h
  src/include/rccl_float8.h
  src/include/rocmwrap.h
  src/include/roctx.h
  src/include/recorder.h
  src/include/scheduler.h
  src/include/shm.h
  src/include/shmutils.h
  src/include/signals.h
  src/include/socket.h
  src/include/strongstream.h
  src/include/sym_kernels.h
  src/include/timer.h
  src/include/transport.h
  src/include/trees.h
  src/include/tuner.h
  src/include/utils.h
  src/include/mlx5/mlx5dvcore.h
  src/include/mlx5/mlx5dvsymbols.h
  src/include/mlx5/mlx5dvwrap.h
  src/include/ionic/ionicdvcore.h
  src/include/ionic/ionicdvsymbols.h
  src/include/ionic/ionicdvwrap.h
  src/include/msccl/msccl_lifecycle.h
  src/include/msccl/msccl_parser.h
  src/include/msccl/msccl_scheduler.h
  src/include/msccl/msccl_setup.h
  src/include/msccl/msccl_status.h
  src/include/msccl/msccl_struct.h
  src/include/nccl_device/comm.h
  src/include/nccl_device/coop.h
  src/include/nccl_device/core.h
  src/include/nccl_device/ll_a2a.h
  src/include/nccl_device/mem_barrier.h
  src/include/nccl_device/ptr.h
  src/include/nccl_device/utility.h
  src/include/nccl_device/impl/comm__funcs.h
  src/include/nccl_device/impl/comm__types.h
  src/include/nccl_device/impl/core__funcs.h
  src/include/nccl_device/impl/core__types.h
  src/include/nccl_device/impl/ll_a2a__funcs.h
  src/include/nccl_device/impl/ll_a2a__types.h
  src/include/nccl_device/impl/mem_barrier__funcs.h
  src/include/nccl_device/impl/mem_barrier__types.h
  src/include/nccl_device/impl/ptr__funcs.h
  src/include/nccl_device/impl/ptr__types.h
  src/include/npkit/npkit.h
  src/include/npkit/npkit_event.h
  src/include/npkit/npkit_struct.h
  src/include/nvtx3/nvToolsExt.h
  src/include/nvtx3/nvToolsExtCounters.h
  src/include/nvtx3/nvToolsExtCuda.h
  src/include/nvtx3/nvToolsExtCudaRt.h
  src/include/nvtx3/nvToolsExtMem.h
  src/include/nvtx3/nvToolsExtMemCudaRt.h
  src/include/nvtx3/nvToolsExtOpenCL.h
  src/include/nvtx3/nvToolsExtPayload.h
  src/include/nvtx3/nvToolsExtPayloadHelper.h
  src/include/nvtx3/nvToolsExtSemanticsCounters.h
  src/include/nvtx3/nvToolsExtSemanticsScope.h
  src/include/nvtx3/nvToolsExtSync.h
  src/include/nvtx3/nvtx3.hpp
  src/include/nvtx3/nvtxDetail/nvtxExtHelperMacros.h
  src/include/nvtx3/nvtxDetail/nvtxExtImpl.h
  src/include/nvtx3/nvtxDetail/nvtxExtImplCounters_v1.h
  src/include/nvtx3/nvtxDetail/nvtxExtImplMem_v1.h
  src/include/nvtx3/nvtxDetail/nvtxExtImplMemCudaRt_v1.h
  src/include/nvtx3/nvtxDetail/nvtxExtImplPayload_v1.h
  src/include/nvtx3/nvtxDetail/nvtxExtInit.h
  src/include/nvtx3/nvtxDetail/nvtxExtPayloadHelperInternal.h
  src/include/nvtx3/nvtxDetail/nvtxExtPayloadTypeInfo.h
  src/include/nvtx3/nvtxDetail/nvtxExtTypes.h
  src/include/nvtx3/nvtxDetail/nvtxImpl.h
  src/include/nvtx3/nvtxDetail/nvtxImplCore.h
  src/include/nvtx3/nvtxDetail/nvtxImplCuda_v3.h
  src/include/nvtx3/nvtxDetail/nvtxImplCudaRt_v3.h
  src/include/nvtx3/nvtxDetail/nvtxImplOpenCL_v3.h
  src/include/nvtx3/nvtxDetail/nvtxImplSync_v3.h
  src/include/nvtx3/nvtxDetail/nvtxInit.h
  src/include/nvtx3/nvtxDetail/nvtxInitDecls.h
  src/include/nvtx3/nvtxDetail/nvtxInitDefs.h
  src/include/nvtx3/nvtxDetail/nvtxLinkOnce.h
  src/include/nvtx3/nvtxDetail/nvtxTypes.h
  src/include/proxy_trace/proxy_trace.h
  src/include/plugin/nccl_net.h
  src/include/plugin/nccl_profiler.h
  src/include/plugin/nccl_tuner.h
  src/include/plugin/plugin.h
  src/include/plugin/net/net_v6.h
  src/include/plugin/net/net_v7.h
  src/include/plugin/net/net_v8.h
  src/include/plugin/net/net_v9.h
  src/include/plugin/net/net_v10.h
  src/include/plugin/net/net_v11.h
  src/include/plugin/profiler/net_ib_v1.h
  src/include/plugin/profiler/net_ib.h
  src/include/plugin/profiler/net_socket_v1.h
  src/include/plugin/profiler/net_socket.h
  src/include/plugin/profiler/profiler_v1.h
  src/include/plugin/profiler/profiler_v2.h
  src/include/plugin/profiler/profiler_v3.h
  src/include/plugin/profiler/profiler_v4.h
  src/include/plugin/profiler/profiler_v5.h
  src/include/plugin/tuner/tuner_v2.h
  src/include/plugin/tuner/tuner_v3.h
  src/include/plugin/tuner/tuner_v4.h
  src/include/plugin/tuner/tuner_v5.h
  src/misc/alt_rsmi.cc
  src/misc/archinfo.cc
  src/misc/argcheck.cc
  src/misc/api_trace.c
  src/misc/api_trace.cc
# src/misc/cudawrap.cc
# src/misc/gdrwrap.cc
  src/misc/ibvsymbols.cc
  src/misc/ibvwrap.cc
  src/misc/ipcsocket.cc
  src/misc/mlx5dvsymbols.cc
  src/misc/mlx5dvwrap.cc
  src/misc/ionicdvsymbols.cc
  src/misc/ionicdvwrap.cc
  src/misc/npkit.cc
# src/misc/nvmlwrap.cc
  src/misc/nvmlwrap_stub.cc
  src/misc/param.cc
  src/misc/rocmwrap.cc
  src/misc/roctx.cc
  src/misc/recorder.cc
  src/misc/shmutils.cc
  src/misc/signals.cc
  src/misc/socket.cc
  src/misc/strongstream.cc
  src/misc/utils.cc
  src/misc/msccl/msccl_lifecycle.cc
  src/misc/msccl/msccl_parser.cc
  src/misc/msccl/msccl_setup.cc
  src/misc/msccl/msccl_status.cc
  src/misc/proxy_trace/proxy_trace.cc
  src/nccl_device/core.cc
  src/nccl_device/ll_a2a.cc
  src/nccl_device/mem_barrier.cc
  src/plugin/net.cc
  src/plugin/plugin_open.cc
  src/plugin/profiler.cc
  src/plugin/tuner.cc
  src/plugin/net/net_v6.cc
  src/plugin/net/net_v7.cc
  src/plugin/net/net_v8.cc
  src/plugin/net/net_v9.cc
  src/plugin/net/net_v10.cc
  src/plugin/net/net_v11.cc
  src/plugin/profiler/profiler_v1.cc
  src/plugin/profiler/profiler_v2.cc
  src/plugin/profiler/profiler_v3.cc
  src/plugin/profiler/profiler_v4.cc
  src/plugin/profiler/profiler_v5.cc
  src/plugin/tuner/tuner_v2.cc
  src/plugin/tuner/tuner_v3.cc
  src/plugin/tuner/tuner_v4.cc
  src/plugin/tuner/tuner_v5.cc
  src/ras/client.cc
  src/ras/client_support.cc
  src/ras/collectives.cc
  src/ras/peers.cc
  src/ras/ras.cc
  src/ras/ras_internal.h
  src/ras/rasnet.cc
  src/register/coll_reg.cc
  src/register/register.cc
  src/register/sendrecv_reg.cc
  src/scheduler/symmetric_sched.cc
  src/transport/coll_net.cc
  src/transport/generic.cc
  src/transport/net.cc
  src/transport/net_ib.cc
  src/transport/net_ib_rocm.cc
  src/transport/net_socket.cc
  src/transport/nvls.cc
  src/transport/p2p.cc
  src/transport/profiler.cc
  src/transport/shm.cc
  src/include/latency_profiler/CollTrace.h
  src/include/latency_profiler/CollTraceEvent.h
  src/include/latency_profiler/CollTraceFunc.h
  src/include/latency_profiler/CollTraceUtils.h
  src/include/latency_profiler/EventQueue.h
  src/misc/latency_profiler/CollTrace.cc
  src/misc/latency_profiler/CollTraceEvent.cc
  src/misc/latency_profiler/CollTraceFunc.cc
  src/misc/latency_profiler/CollTraceUtils.cc
)

if(USE_AMDSMI)
  set(SMI_SOURCES
    src/include/amdsmi_wrap.h
    src/misc/amdsmi_wrap.cc
  )
else()
  set(SMI_SOURCES
    src/include/rocm_smi_wrap.h
    src/misc/rocm_smi_wrap.cc
  )
endif()
list(APPEND SRC_FILES ${SMI_SOURCES})

if (ENABLE_MSCCL_KERNEL)
  set(MSCCL_KERNEL_SOURCES
    src/device/msccl_kernel_impl.h
    src/include/msccl/msccl_kernel.h
  )
  list(APPEND SRC_FILES ${MSCCL_KERNEL_SOURCES})
endif()

if (ENABLE_MSCCLPP)
  set(MSCCLPP_SOURCES
    src/include/mscclpp/mscclpp_nccl.h
    src/misc/mscclpp/mscclpp_nccl.cc
  )
  list(APPEND SRC_FILES ${MSCCLPP_SOURCES})
endif()

# Hipify source files (copy of source generated into hipify directory)
#==================================================================================================
find_program(hipify-perl_executable hipify-perl)
if(NOT hipify-perl_executable)
  message(FATAL_ERROR "hipify-perl not found")
endif()
set(HIPIFY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipify")

## Loop over each source file to hipify
foreach(SRC_FILE ${SRC_FILES})
  # Check that file exists
  if (NOT EXISTS ${CMAKE_SOURCE_DIR}/${SRC_FILE})
    message(FATAL_ERROR "Unable to find file listed in CMakeLists.txt: ${CMAKE_SOURCE_DIR}/${SRC_FILE}")
  endif()

  # Establish hipified copy of the source file
  set(HIP_FILE "${HIPIFY_DIR}/${SRC_FILE}")
  get_filename_component(HIP_FILE_DIR ${HIP_FILE} DIRECTORY)

  # Make sure the file name is unique and there is no duplicate
  add_file_unique(HIP_SOURCES ${HIP_FILE})

  # Convert .cu files to .cpp so that they get processed properly
  string(REPLACE "\.cuh" "\.h" HIP_FILE ${HIP_FILE})
  string(REPLACE "\.cu" "\.cu.cpp" HIP_FILE ${HIP_FILE})
  list(APPEND HIP_SOURCES ${HIP_FILE})

  # Create a custom command to create hipified source code
  if (FAULT_INJECTION)
    add_custom_command(
      OUTPUT ${HIP_FILE}
      COMMAND mkdir -p ${HIP_FILE_DIR}
              && ${hipify-perl_executable} -quiet-warnings ${CMAKE_SOURCE_DIR}/${SRC_FILE} -o ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/add_unroll.sh ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/add_faults.sh ${HIP_FILE}
      MAIN_DEPENDENCY ${SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
    )
  else()
    add_custom_command(
      OUTPUT ${HIP_FILE}
      COMMAND mkdir -p ${HIP_FILE_DIR}
              && ${hipify-perl_executable} -quiet-warnings ${CMAKE_SOURCE_DIR}/${SRC_FILE} -o ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/add_unroll.sh ${HIP_FILE}
      MAIN_DEPENDENCY ${SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
    )
  endif()
endforeach()

# Adding custom target to hipify all the source files
# This is required to make sure that all the hipified source files are
# available before compiling the unit tests executable(s)
add_custom_target(hipify_all DEPENDS ${HIP_SOURCES})

# Generate device/host tables and all the collective functions that are going to be in librccl.so
#==================================================================================================
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if (NOT Python3_FOUND)
  message(FATAL_ERROR "RCCL requires Python3 for generating host/device tables")
endif()

set(GEN_DIR "${HIPIFY_DIR}/gensrc")
set(GEN_SYM_DIR "${GEN_DIR}/symmetric")

if(ONLY_FUNCS)
  message(WARNING "Using ONLY_FUNCS = ${ONLY_FUNCS}. Not meant for release builds.")
endif()

# Execute the python script to generate required collective functions
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/src/device/generate.py ${GEN_DIR} ${IFC_ENABLED} ${COLLTRACE} ${ENABLE_MSCCL_KERNEL} ${BUILD_LOCAL_GPU_TARGET_ONLY} ${ONLY_FUNCS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE gen_py_result
    ERROR_VARIABLE gen_py_error
)
if (gen_py_result)
  message(SEND_ERROR "Error: ${gen_py_error}")
  message(FATAL_ERROR "${CMAKE_SOURCE_DIR}/src/device/generate.py failed")
endif()

if (GENERATE_SYM_KERNELS)
  # Execute the python script to generate required symmetric memory kernels
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/src/device/symmetric/generate.py ${GEN_SYM_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE gen_sym_py_result
    ERROR_VARIABLE gen_sym_py_error
  )
  if (gen_sym_py_result)
    message(SEND_ERROR "Error: ${gen_sym_py_error}")
    message(FATAL_ERROR "${CMAKE_SOURCE_DIR}/src/device/symmetric/generate.py failed")
  endif()
endif()

# Find the generated files in the output directory
file(GLOB_RECURSE GENERATED_FILES "${GEN_DIR}/*")

# Append all found generated files to the list
foreach(file ${GENERATED_FILES})
  list(APPEND HIP_SOURCES ${file})
endforeach()

# Create an initial git_version.cpp file (that will be updated with latest git version)
#==================================================================================================
# Create initial empty file at configure time
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp "")

# Add a custom target that always runs at build time to update git version
add_custom_target(update_git_version
  ALL
  COMMAND ${CMAKE_COMMAND} -DRCCL_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DRCCL_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/git_version.cmake
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp
  COMMENT "Updating git version information"
  VERBATIM
)

list(APPEND HIP_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp)

# Set up RCCL library
#==================================================================================================
## Set RCCL source files
add_library(rccl ${HIP_SOURCES})

## Set RCCL dependencies
## Ensure git version is updated before building rccl
add_dependencies(rccl update_git_version)

## Set RCCL include directories
target_include_directories(rccl PRIVATE ${PROJECT_BINARY_DIR}/include)        # for generated rccl.h header
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src)                    # for hipfied headers
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/device)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/device/network/unpack)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/mlx5)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/nccl_device)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/ionic)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/plugin)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/gensrc)
target_include_directories(rccl PRIVATE ${HSA_INCLUDE_PATH})
target_include_directories(rccl PRIVATE ${ROCM_SMI_INCLUDE_DIR})
target_include_directories(rccl PRIVATE ${ROCMCORE_PATH}/include)
if(DEMANGLE_DIR)
  target_include_directories(rccl PRIVATE ${DEMANGLE_DIR})
endif()
if(ROCTX_ENABLE)
  target_include_directories(rccl PRIVATE ${ROCTRACER_INCLUDE_DIR})
endif()


## Set RCCL compile definitions
if(COLLTRACE)
  target_compile_definitions(rccl PRIVATE ENABLE_COLLTRACE)
endif()
if(ENABLE_MSCCL_KERNEL)
  message(WARNING "MSCCL is deprecated and will be removed in a future version of RCCL.")
  target_compile_definitions(rccl PRIVATE COMPILE_MSCCL_KERNEL)
endif()
if(ENABLE_MSCCLPP)
  target_compile_definitions(rccl PRIVATE ENABLE_MSCCLPP)
endif()

if(USE_AMDSMI)
  target_compile_definitions(rccl PRIVATE USE_AMDSMI)
else()
  if(HAVE_ROCM_SMI64CONFIG)
    target_compile_definitions(rccl PRIVATE USE_ROCM_SMI64CONFIG)
  endif()
  if(HAVE_ROCM_SMI_THREAD_ONLY_MUTEX)
    target_compile_definitions(rccl PRIVATE USE_ROCM_SMI_THREAD_ONLY_MUTEX)
  endif()
endif()
if(ENABLE_WARP_SPEED)
  target_compile_definitions(rccl PRIVATE ENABLE_WARP_SPEED)
endif()
if(ENABLE_ROCSHMEM)
  target_compile_definitions(rccl PRIVATE ENABLE_ROCSHMEM)
endif()

# ==== rocSHMEM integration (optional) ====

if (ENABLE_ROCSHMEM)
  add_rocshmem_targets()
  # Ensure rocSHMEM is fully built/installed before compiling rccl
  if (TARGET rocshmem_ext)
    add_dependencies(rccl rocshmem_ext)
  endif()

  if (ROCSHMEM_INCLUDE_DIR)
    target_include_directories(rccl PRIVATE ${ROCSHMEM_INCLUDE_DIR})
  endif()

  # Moved to where MSCCL target_links
  ## target_link_libraries(rccl PRIVATE ${ROCSHMEM_LIBRARY})
  target_link_libraries(rccl PRIVATE ${IBVERBS})

endif()

# NPKit flags
## May be better to move these to a separate file
if(ENABLE_NPKIT)
  message(WARNING "NPKit is deprecated and will be removed in a future version of RCCL. Please consider using alternative profiling tools.")
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_RECV_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_RECV_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_RECV_REDUCE_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_SEND_FROM_OUTPUT_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_SEND_FROM_OUTPUT_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_REDUCE_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_REDUCE_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_FROM_OUTPUT_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_FROM_OUTPUT_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_SIMPLE_WAIT_PEER_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_SIMPLE_WAIT_PEER_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL_WAIT_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL_WAIT_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL128_WAIT_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL128_WAIT_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_NET_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_NET_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_NET_TEST_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_NET_TEST_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_NET_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_NET_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_RECV_LOCAL_COPY_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_RECV_LOCAL_COPY_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_RECV_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_RECV_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_RECV_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_RECV_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_REDUCE_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_REDUCE_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_RUN_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_RUN_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_INIT_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_INIT_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_BROADCAST_RING_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_BROADCAST_RING_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_COPY_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_COPY_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME)
endif()

if(PROFILE)
  target_compile_definitions(rccl PRIVATE ENABLE_PROFILING)
endif()
if(ROCTX_ENABLE)
  target_compile_definitions(rccl PRIVATE ROCTX_ENABLE)
else()
  target_compile_definitions(rccl PRIVATE NVTX_NO_IMPL)
  target_compile_definitions(rccl PRIVATE NVTX_DISABLE)
endif()
if(TRACE)
  target_compile_definitions(rccl PRIVATE ENABLE_TRACE)
endif()
if(${HIP_CONTIGUOUS_MEMORY})
  target_compile_definitions(rccl PRIVATE HIP_CONTIGUOUS_MEMORY)
  message(STATUS "HIP_CONTIGUOUS_MEMORY enabled")
else()
  message(STATUS "HIP_CONTIGUOUS_MEMORY disabled")
endif()
if("${hip_version_string}" VERSION_GREATER_EQUAL "5.7.31920")
  target_compile_definitions(rccl PRIVATE HIP_UNCACHED_MEMORY)
  message(STATUS "HIP_UNCACHED_MEMORY enabled")
else()
  message(STATUS "HIP_UNCACHED_MEMORY disabled - requires HIP version >= 5.7.31920")
  # keep --hipcc-func-supp on older HIP and compiler
  if(NOT IFC_ENABLED)
    target_compile_options(rccl PRIVATE --hipcc-func-supp)
    message(STATUS "--hipcc-func-supp enabled")
  else()
    message(STATUS "--hipcc-func-supp disabled")
  endif()
endif()
if (HIP_HOST_UNCACHED_MEMORY)
  target_compile_definitions(rccl PRIVATE HIP_HOST_UNCACHED_MEMORY)
  message(STATUS "HIP_HOST_UNCACHED_MEMORY enabled")
else()
  message(STATUS "HIP_HOST_UNCACHED_MEMORY disabled")
endif()
if (BUILD_BFD)
  if (HAVE_BFD)
    target_compile_definitions(rccl PRIVATE HAVE_BFD)
  endif()
  if (HAVE_DECL_BFD_GET_SECTION_FLAGS)
    target_compile_definitions(rccl PRIVATE HAVE_DECL_BFD_GET_SECTION_FLAGS)
  endif()
  if (HAVE_DECL_BFD_GET_SECTION_VMA)
    target_compile_definitions(rccl PRIVATE HAVE_DECL_BFD_GET_SECTION_VMA)
  endif()
  if (HAVE_TWO_ARG_BFD_SECTION_SIZE)
    target_compile_definitions(rccl PRIVATE HAVE_TWO_ARG_BFD_SECTION_SIZE)
  endif()
endif()
if (IFC_ENABLED)
  target_compile_definitions(rccl PRIVATE USE_INDIRECT_FUNCTION_CALL)
endif()
if(DEMANGLE_DIR)
  target_compile_definitions(rccl PRIVATE "HAVE_CPLUS_DEMANGLE=1")
  target_compile_definitions(rccl PRIVATE "HAVE_DECL_BASENAME=1")
endif()
if(LL128_ENABLED)
  target_compile_definitions(rccl PRIVATE ENABLE_LL128)
endif()
if (GENERATE_SYM_KERNELS)
  target_compile_definitions(rccl PRIVATE GENERATE_SYM_KERNELS)
endif()

## Set RCCL compile options
if (HAVE_PARALLEL_JOBS)
  target_compile_options(rccl PRIVATE -parallel-jobs=12)
endif()

if (ROCM_VERSION VERSION_GREATER_EQUAL "60200")
  target_compile_options(rccl PRIVATE --offload-compress)    # Compress GPU code at compile time.
  target_link_libraries(rccl PRIVATE --offload-compress)     # Compress GPU code at link time.
  message(STATUS "--offload-compress enabled - ROCm version >= 6.2.0")
else()
  message(STATUS "--offload-compress disabled - ROCm version < 6.2.0")
endif()

target_compile_options(rccl PRIVATE -Werror=uninitialized)
target_compile_options(rccl PRIVATE -Werror=sometimes-uninitialized)
target_compile_options(rccl PRIVATE -Wall)
target_compile_options(rccl PRIVATE -Werror=deprecated-copy-with-user-provided-copy)
target_compile_options(rccl PRIVATE -Wno-format-nonliteral)
target_compile_options(rccl PRIVATE -Wno-unused-function)
target_compile_options(rccl PRIVATE -fgpu-rdc)

if(QUIET_WARNINGS)
  target_compile_options(rccl PRIVATE -Wno-invalid-offsetof)
  target_compile_options(rccl PRIVATE -Wno-unused-result)
  target_compile_options(rccl PRIVATE -Wno-macro-redefined)
  target_compile_options(rccl PRIVATE -Wno-unused-label)
  target_compile_options(rccl PRIVATE -Wno-unused-variable)
  target_compile_options(rccl PRIVATE -Wno-unused-private-field)
  target_compile_options(rccl PRIVATE -Wno-null-conversion)
  target_compile_options(rccl PRIVATE -Wno-missing-braces)
endif()

## Set RCCL compile and linker options for unit tests and code coverage
if(ENABLE_CODE_COVERAGE)
  if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    message(FATAL_ERROR "Code coverage is enabled, but the build type is '${CMAKE_BUILD_TYPE}'. "
                        "Code coverage requires 'Debug' build types to expose internal symbols. "
                        "Please set CMAKE_BUILD_TYPE to 'Debug' and reconfigure.")
  endif()

  message(STATUS "Code coverage is enabled with build type '${CMAKE_BUILD_TYPE}'.")

  target_compile_options(rccl PRIVATE
    -fvisibility=default -Xarch_host -fprofile-instr-generate -Xarch_host -fcoverage-mapping)

  set(COVERAGE_SHARED_LINKER_FLAGS
    -fprofile-generate
    -Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ORIGIN
  )

  set(COVERAGE_EXE_LINKER_FLAGS
    -fprofile-generate
    -Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ORIGIN/../lib
  )

  target_link_options(rccl PRIVATE ${COVERAGE_SHARED_LINKER_FLAGS})
  target_link_options(rccl PRIVATE ${COVERAGE_EXE_LINKER_FLAGS})
elseif(BUILD_TESTS) # Enable default/hidden visibility based on build type and ROCM_VERSION
  if (ROCM_VERSION VERSION_GREATER_EQUAL "60400" AND CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_options(rccl PRIVATE -fvisibility=default)
  else()
    target_compile_options(rccl PRIVATE -fvisibility=hidden)
  endif()
else()  # Enable hidden visibility for library without tests/code coverage enabled
  target_compile_options(rccl PRIVATE -fvisibility=hidden)
endif()

if (HAVE_KERNARG_PRELOAD)
  target_compile_options(rccl PRIVATE -mllvm --amdgpu-kernarg-preload-count=16)
endif()

if (REPORT_KERNEL_RESOURCE_USE)
  target_link_options(rccl PRIVATE -Rpass-analysis=kernel-resource-usage)
endif()

if (DUMP_ASM) # Save temporary files from kernel compilation
  message(STATUS "Disassembling librccl.so to asm")
  # Maintain symbols but without changing code.  Keep additional data in dwarf section of binary.
  target_compile_options(rccl PRIVATE -gline-tables-only)
  set(OBJ_DUMP ${ROCM_PATH}/llvm/bin/llvm-objdump)

  add_custom_command(TARGET rccl POST_BUILD
    COMMENT "Disassembling RCCL library"
    COMMAND /bin/bash -c "${OBJ_DUMP} --offload-fatbin librccl.so"
    VERBATIM
  )
  foreach(GPUARCH ${GPU_TARGETS})
    add_custom_command(TARGET rccl POST_BUILD
        COMMENT "Disassembling RCCL library to dump assembly for ${GPUARCH}"
        COMMAND /bin/bash -c "${OBJ_DUMP} -d -l --source --symbolize-operands librccl.so.0.hipv4-amdgcn-amd-amdhsa--${GPUARCH} > librccl.${GPUARCH}.s"
        VERBATIM
    )
  endforeach()
endif()

## NOTE: This is currently being handled by rocm-cmake, however may need to be re-enabled in the future
#foreach(target ${GPU_TARGETS})
#  target_compile_options(rccl PRIVATE --offload-arch=${target})
#endforeach()

if(BUILD_ADDRESS_SANITIZER)
  target_compile_options(rccl PRIVATE -fsanitize=address -shared-libasan)
endif()
if(TIMETRACE)
  target_compile_options(rccl PRIVATE -ftime-trace)
endif()
if (FAULT_INJECTION)
  target_compile_definitions(rccl PRIVATE ENABLE_FAULT_INJECTION)
  message(STATUS "Fault injection enabled")
endif()

## Set RCCL linked library directories
target_link_directories(rccl PRIVATE ${SMI_LIB_DIR})

if (ROCM_VERSION VERSION_GREATER_EQUAL "60100")
    option(RCCL_ROCPROFILER_REGISTER "Enable rocprofiler-register support" ON)
else()
    if(RCCL_ROCPROFILER_REGISTER)
        message(AUTHOR_WARNING "RCCL_ROCPROFILER_REGISTER is not valid option for ROCm < 6.2. Current ROCm version: ${ROCM_VERSION}")
    endif()
    set(RCCL_ROCPROFILER_REGISTER OFF CACHE BOOL "" FORCE)
endif()
if(RCCL_ROCPROFILER_REGISTER)
  find_package(rocprofiler-register REQUIRED)
  target_compile_definitions(rccl PRIVATE RCCL_ROCPROFILER_REGISTER=1)
  target_link_libraries(
      rccl PRIVATE rocprofiler-register::rocprofiler-register)
endif()

## Set RCCL linked libraries
if (HAVE_BFD)
  target_link_libraries(rccl PRIVATE bfd)
  if(HAVE_IBERTY)
    target_link_libraries(rccl PRIVATE iberty z)
  endif()
endif()
if (ROCTX_ENABLE)
  target_link_libraries(rccl PRIVATE ${ROCTX_LIB})
endif()
target_link_libraries(rccl PRIVATE   -fgpu-rdc)             # Required when linking relocatable device code
target_link_libraries(rccl PRIVATE   Threads::Threads)
target_link_libraries(rccl INTERFACE hip::host)
target_link_libraries(rccl PRIVATE   hip::device)
target_link_libraries(rccl PRIVATE   dl)
target_link_libraries(rccl PRIVATE   ${SMI_LIBRARIES})
target_link_libraries(rccl PRIVATE fmt::fmt-header-only)
if(ENABLE_MSCCLPP)
  target_link_libraries(rccl PRIVATE mscclpp_nccl)
endif()
if(ENABLE_ROCSHMEM)
  target_link_libraries(rccl PRIVATE ${ROCSHMEM_LIBRARY})
  target_link_libraries(rccl PRIVATE ${IBVERBS})
endif()

## Set RCCL link options
## Find out available memory
execute_process(
  COMMAND bash "-c" "cat /sys/fs/cgroup/memory.max"
  OUTPUT_VARIABLE memory_max_string)
if (${memory_max_string} MATCHES "^[0-9]+")
  math(EXPR memory_in_gb "${memory_max_string} / (1024 * 1024 * 1024)")
else()
  execute_process(
    COMMAND bash "-c" "free | grep -o '[[:digit:]]*' | head -1"
    OUTPUT_VARIABLE memory_max_string)
  ## memory_max_string holds the free memory in KB
  if (${memory_max_string} MATCHES "^[0-9]+")
    math(EXPR memory_in_gb "${memory_max_string} / (1024 * 1024)") ## KB to GB conversion
  else()
    cmake_host_system_information(RESULT memory_max_string QUERY AVAILABLE_PHYSICAL_MEMORY )
    math(EXPR memory_in_gb "${memory_max_string} / 1024")
  endif()
endif()
## Reserve 16GB for each linker job. Limit max number of linker jobs to 16
if (HAVE_PARALLEL_JOBS)
  math(EXPR num_linker_jobs "(${memory_in_gb} + 15) / 16")
  if (${num_linker_jobs} GREATER_EQUAL "16")
    set(num_linker_jobs "16")
  endif()
  message(STATUS "Use ${num_linker_jobs} jobs for linking")
  target_link_options(rccl PRIVATE -parallel-jobs=${num_linker_jobs})       # Use multiple threads to link
endif()
if(BUILD_ADDRESS_SANITIZER)
  target_link_options(rccl PRIVATE -fuse-ld=lld)
endif()
if(TIMETRACE)
  target_link_options(rccl PRIVATE -ftime-trace)
endif()

if(NOT BUILD_SHARED_LIBS)
  message(STATUS "Building static RCCL library")
else()
  message(STATUS "Building shared RCCL library")
endif()
if (HAVE_KERNARG_PRELOAD)
  target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker -mllvm=-amdgpu-kernarg-preload-count=16")
endif()

if(ENABLE_MSCCLPP)
  add_mscclpp_targets()
endif()

## Track linking time
set_property(TARGET rccl PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

## Setup librccl.so version
rocm_set_soversion(rccl "1.0")

if(NOT BUILD_SHARED_LIBS)
  # To create a static lib with `-fgpu-rdc`, you need `--emit-static-lib` and `--hip-link`.
  # You also need to invoke amdclang++ again to trigger GPU code generation.
  set(static_link_flags
    ${CXXFLAGS}
    --hip-link
    -fgpu-rdc
    --emit-static-lib
  )

  # Find all the libraries we need to link at link time to include them in the clang link
  # command line.
  get_target_property(rccl_libs rccl LINK_LIBRARIES)
  foreach(target ${rccl_libs})
    if(TARGET ${target})
      get_target_property(location ${target} LOCATION)
      if(location)
        LIST(APPEND static_link_flags -l${location})
      endif()
    endif()
  endforeach()

  foreach(target ${GPU_TARGETS})
    list(APPEND static_link_flags --offload-arch=${target})
  endforeach()
  list(JOIN static_link_flags " " flags_str)

  # Invoking amdclang++ this way will produce a static archive, so just override ARCHIVE_CREATE.
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_CXX_COMPILER> ${flags_str} -o <TARGET> <OBJECTS>")
endif()

# Install settings
#==================================================================================================
## Specify install targets
rocm_install_targets(TARGETS rccl)
rocm_install(FILES       ${PROJECT_BINARY_DIR}/include/rccl/rccl.h src/include/plugin/nccl_net.h
             DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rccl)
rocm_install(FILES       src/include/api_trace.h
             DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rccl/amd_detail)
file(COPY tools/msccl-algorithms DESTINATION ${PROJECT_BINARY_DIR})
file(COPY tools/msccl-unit-test-algorithms DESTINATION ${PROJECT_BINARY_DIR})
## Install Algorithm files under share folder
rocm_install(DIRECTORY ${PROJECT_BINARY_DIR}/msccl-algorithms DESTINATION ${CMAKE_INSTALL_DATADIR}/rccl)
rocm_install(DIRECTORY ${PROJECT_BINARY_DIR}/msccl-unit-test-algorithms DESTINATION ${CMAKE_INSTALL_DATADIR}/rccl)

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

## Building RCCL RAS
include(cmake/rcclRAS.cmake)

if(BUILD_TESTS)
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(tests PACKAGE_NAME unittests)
  add_subdirectory(test)

  if(BUILD_SHARED_LIBS)
    add_custom_command(TARGET rccl POST_BUILD
      COMMENT "Extracting metadata from librccl.so"
      COMMAND COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/extract_metadata.cmake
      VERBATIM
    )
  endif()
endif()

rocm_create_package(
  NAME        rccl
  DESCRIPTION "ROCm Communication Collectives Library"
  MAINTAINER  "RCCL Maintainer <rccl-maintainer@amd.com>"
  LDCONFIG)
