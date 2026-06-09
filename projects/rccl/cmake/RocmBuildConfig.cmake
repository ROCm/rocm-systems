# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# RCCL ROCm / HIP build configuration.
#==================================================================================================
# This module is extracted (verbatim) from the top-level CMakeLists.txt so that the
# root file mirrors the section structure of upstream NCCL's CMakeLists.txt. It
# bundles the RCCL-specific build setup that has no NCCL counterpart: GPU target
# selection, ROCm toolchain discovery, dependency/feature probing, and the
# SMI / BFD / ROCTX detection logic.
#
# IMPORTANT: this file is pulled in with include() (NOT add_subdirectory), so it
# executes in the top-level directory scope. Every variable it sets (GPU_TARGETS,
# ROCM_VERSION, SMI_*, HSA_INCLUDE_PATH, IFC_ENABLED, LL128_ENABLED, CE_ENABLED,
# HIP_*_MEMORY, ROCTX_ENABLE, ...) therefore stays visible to the rest of the
# top-level CMakeLists.txt and to add_subdirectory(src), exactly as before.
#==================================================================================================

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
      gfx1151
      gfx1200
      gfx1201
      gfx1250)

# Load CMake modules
#==================================================================================================
include(CheckIncludeFiles)
include(CheckIncludeFileCXX)
include(CheckSymbolExists)
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
include(GNUInstallDirs)
include(${CMAKE_CURRENT_LIST_DIR}/Dependencies.cmake) # GTest, rocm-cmake, rocm_local_targets
include(${CMAKE_CURRENT_LIST_DIR}/CheckSymbolExistsNoWarn.cmake)
# Include rocSHMEM build module only if enabled
if(ENABLE_ROCSHMEM)
  include(${CMAKE_CURRENT_LIST_DIR}/ROCSHMEM.cmake)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(ENABLE_COMPRESS                         "Enable GPU code compression"                   ON)
rccl_apply_nccl_compat_translations()

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
include(${CMAKE_CURRENT_LIST_DIR}/rocmIb.cmake)

# Modify GPU architectures for Address Sanitizer builds by appending "xnack+"
if (BUILD_ADDRESS_SANITIZER)
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libclang_rt.asan-x86_64.so
    OUTPUT_VARIABLE ASAN_RUNTIME_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  get_filename_component(ASAN_RUNTIME_DIR "${ASAN_RUNTIME_PATH}" DIRECTORY)
  message(STATUS "ASAN runtime directory: ${ASAN_RUNTIME_DIR}")

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

# Fail fast if we ended up with no GPU targets at all
if(NOT GPU_TARGETS OR "${GPU_TARGETS}" STREQUAL "NOTFOUND")
  message(FATAL_ERROR
    "GPU_TARGETS is empty (resolved to '${GPU_TARGETS}').\n"
    "This usually means ./install.sh -l was used on a host with no visible "
    "AMD GPU (no /dev/kfd, no /dev/dri), so local-arch detection found "
    "nothing to target.\n"
    "\n"
    "If you wish to build here anyway, re-run with an explicit GPU "
    "architecture, e.g.:\n"
    "    ./install.sh --amdgpu_targets gfx1201\n")
endif()

message(STATUS "Compiling for ${GPU_TARGETS}")

## NOTE: Reload rocm-cmake in order to update GPU_TARGETS
include(${CMAKE_CURRENT_LIST_DIR}/Dependencies.cmake) # Reloading to use desired GPU_TARGETS instead of defaults

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
set(CMAKE_CXX_EXTENSIONS OFF)  # Without this line, it will add -std=gnu++20 instead, which has some issues.
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

### Check for hipMemImportFromShareableHandle function support
check_symbol_exists("hipMemImportFromShareableHandle" "hip/hip_runtime_api.h" HIP_FABRIC_API_FUNC)

### Check for hipMemFabricHandle_t type (required for fabric API)
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
  #include <hip/hip_runtime_api.h>
  int main() {
    hipMemFabricHandle_t handle;
    (void)handle;
    return 0;
  }
" HIP_FABRIC_HANDLE_TYPE)

### Check for hipMemcpyBatchAsync support (Copy Engine batch API)
check_symbol_exists("hipMemcpyBatchAsync" "hip/hip_runtime_api.h" HIP_MEMCPY_BATCH_ASYNC)

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

## Check for Copy Engine (CE) batch API support
## hipMemcpyBatchAsync symbol presence is the sole gate: if the SDK exposes the
## symbol, CE is supported regardless of the exact version string.
if(HIP_MEMCPY_BATCH_ASYNC)
  set(CE_ENABLED ON)
  message(STATUS "Copy Engine (CE) batch API enabled")
else()
  set(CE_ENABLED OFF)
  message(STATUS "Copy Engine (CE) batch API disabled - hipMemcpyBatchAsync not found in hip/hip_runtime_api.h")
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

    ### Check whether the installed amdsmi headers expose the UALoE fabric struct
    ### (amdsmi_fabric_info_t). The probe compiles against the struct only — no
    ### link-time dependency on amdsmi_get_gpu_fabric_info — because amdsmi is
    ### loaded lazily at runtime via dlopen. If the struct is absent, the compat
    ### types in amdsmi_wrap.h must be compiled in.
    set(CMAKE_REQUIRED_INCLUDES "${SMI_INCLUDE_DIR}")
    check_cxx_source_compiles("
      #include <amd_smi/amdsmi.h>
      int main() {
        amdsmi_fabric_info_t info{};
        (void)info;
        return 0;
      }
    " AMDSMI_FABRIC_API)
    unset(CMAKE_REQUIRED_INCLUDES)
    if(AMDSMI_FABRIC_API)
      message(STATUS "amdsmi UALoE fabric API found - enabling AMDSMI_FABRIC_DIRECT")
    else()
      message(STATUS "amdsmi UALoE fabric API not found - using compat types")
    endif()
  endif()
endif()

set(USE_AMDSMI ON CACHE INTERNAL "Force enable AMD SMI path since we use lazy loading, and rsmi path is available as fallback if AMD SMI is not enabled at runtime")

## Commenting this out for full deprecation in a future PR
# if(NOT USE_AMDSMI)
#   ## Fallback to rocm-smi if amd-smi not found or ROCm < 7.11.0
#   message(WARNING "Could not find amd_smi. Falling back to rocm_smi.")
#   find_package(rocm_smi PATHS ${ROCM_PATH}/lib/cmake/rocm_smi)
#   if(rocm_smi_FOUND)
#     set(SMI_INCLUDE_DIR "${rocm_smi_INCLUDE_DIR}" CACHE INTERNAL "rocm-smi include directory")
#     set(SMI_LIB_DIR "${rocm_smi_LIB_DIR}" CACHE INTERNAL "rocm-smi library directory")
#   else()
#     message(WARNING "CMake could not find rocm-smi. Checking old include directory structure for rocm_smi")
#     set(SMI_INCLUDE_DIR "${ROCM_PATH}/rocm_smi/include")
#     set(SMI_LIB_DIR     "${ROCM_PATH}/rocm_smi/lib")
#   endif()

#   if(NOT EXISTS "${SMI_INCLUDE_DIR}" OR NOT EXISTS "${SMI_LIB_DIR}")
#     message(FATAL_ERROR "rocm_smi not found in ${SMI_INCLUDE_DIR}")
#   endif()
#   message(STATUS "Found rocm_smi at ${SMI_INCLUDE_DIR}")
#   set(SMI_LIB_NAME "rocm-smi-lib" CACHE INTERNAL "rocm-smi-lib for packaging")
#   set(SMI_LIBRARIES rocm_smi64)

#   check_include_file_cxx("${SMI_INCLUDE_DIR}/rocm_smi/rocm_smi64Config.h" HAVE_ROCM_SMI64CONFIG)

#   ### Check for RSMI_INIT_FLAG_THRAD_ONLY_MUTEX support
#   file(READ "${SMI_INCLUDE_DIR}/rocm_smi/rocm_smi.h" rocm_smi_incl)
#   string(FIND "${rocm_smi_incl}" "RSMI_INIT_FLAG_THRAD_ONLY_MUTEX" matchres)
#   if(${matchres} EQUAL -1)
#     message(STATUS "RSMI_INIT_FLAG_THRAD_ONLY_MUTEX not supported")
#   else()
#     message(STATUS "RSMI_INIT_FLAG_THRAD_ONLY_MUTEX supported")
#     set(HAVE_ROCM_SMI_THREAD_ONLY_MUTEX True)
#   endif ()
# endif()

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
