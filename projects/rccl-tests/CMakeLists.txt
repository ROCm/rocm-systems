# Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.

# CMake version minimum requirements
#==================================================================================================
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# CMake Toolchain file to define compilers and path to ROCm
#==================================================================================================
if (NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/toolchain-linux.cmake")
  message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# RCCL Tests project
#==================================================================================================
project(rccl-tests LANGUAGES CXX)

enable_testing()

# Build options
#==================================================================================================
option(USE_MPI                     "Build RCCL-tests with MPI support."           OFF)
option(ENABLE_DEVICE_API           "Build with NCCL Device API support (GIN)."    OFF)
option(ENABLE_ROCSHMEM_GIN         "Build with rocshmem GIN backend support."     OFF)
option(BUILD_LOCAL_GPU_TARGET_ONLY "Build only for GPUs detected on this machine" OFF)
option(BUILD_RCCL_TESTS_HOST_UNIT_TESTS
       "Build CPU-only host unit tests under test/ (standalone builds only)"   ON)

if (NOT CMAKE_BUILD_TYPE)
  message(WARNING "CMAKE_BUILD_TYPE is not defined. Setting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type")
endif()

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

# Get additional packages required
include(CheckIncludeFiles)
include(CheckSymbolExists)
include(cmake/Dependencies.cmake) # rocm-cmake, rocm_local_targets
include(cmake/CheckSymbolExistsNoWarn.cmake)

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
else()
  message(FATAL_ERROR "RCCL-Tests can be built only with amdclang++")
endif()

# Set CMAKE flags
#==================================================================================================
set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "")
set(CMAKE_CXX_STANDARD   17)   # rocshmem GIN QP device headers require C++17
set(CMAKE_CXX_EXTENSIONS OFF)  # Without this line, it will add -std=gnu++14 instead, which has some issues.
set(CPACK_PACKAGING_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Path to install to when packaged.")
if(ROCM_PATH)
  #list(APPEND CMAKE_PREFIX_PATH  # Temporary workaround
  list(PREPEND CMAKE_PREFIX_PATH  # Add ROCM_PATH to CMake search paths (for finding HIP / HSA
              ${ROCM_PATH}
              ${ROCM_PATH}/hip
              ${ROCM_PATH}/llvm)
endif()

# Check for required dependencies
#==================================================================================================
## Check for Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

## Check for HIP
# HIP_PLATFORM fails to auto detect in some ROCK builds
if (NOT DEFINED HIP_PLATFORM AND NOT DEFINED ENV{HIP_PLATFORM})
  set(HIP_PLATFORM "amd" CACHE STRING "HIP platform (amd or nvidia)")
endif()
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
# Use hip_VERSION from find_package(hip) above
string(REGEX MATCH "^([0-9]+\\.[0-9]+\\.[0-9]+)" hip_version_string "${hip_VERSION}")
message(STATUS "${COMPILER_EXE_NAME} HIP version:    ${hip_version_string}")

##Check for ROCm version
set(EXPLICIT_ROCM_VERSION "" CACHE STRING "Explicit ROCM version to compile to (auto detect if empty)")
if(EXPLICIT_ROCM_VERSION)
  set(rocm_version_string "${EXPLICIT_ROCM_VERSION}")
elseif(ROCM_PATH)
  find_file(_rocm_version_file "version" PATH_SUFFIXES ".info" "core/.info"
    HINTS ${ROCM_PATH} PATHS /opt/rocm NO_DEFAULT_PATH)
  if(NOT _rocm_version_file)
    message(FATAL_ERROR "Could not find ROCm version file under ${ROCM_PATH}")
  endif()
  message(STATUS "Reading ROCM version from ${_rocm_version_file}")
  file(READ "${_rocm_version_file}" rocm_version_string)
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

## Check for RCCL
find_package(RCCL CONFIG REQUIRED HINTS "${CMAKE_PREFIX_PATH}" PATHS "${ROCM_PATH}")
if (RCCL_FOUND)
    message(STATUS "RCCL version : ${RCCL_VERSION}")
    message(STATUS "RCCL include path : ${RCCL_INCLUDE_DIRS}")
    message(STATUS "RCCL libraries : ${RCCL_LIBRARIES}")
endif()

## Check for MPI (if enabled)
if (USE_MPI)
    find_package(MPI REQUIRED)
    if (MPI_FOUND)
        message(STATUS "MPI include path : ${MPI_CXX_INCLUDE_PATH}")
        message(STATUS "MPI libraries : ${MPI_CXX_LIBRARIES}")
        add_definitions(-DMPI_SUPPORT)
    else()
        message ("-- no MPI library found")
    endif()
else()
    message ("-- MPI support disabled")
endif()

if (ENABLE_DEVICE_API)
    message(STATUS "NCCL Device API (GIN) support enabled")
    add_definitions(-DENABLE_DEVICE_API)
endif()

if (ENABLE_ROCSHMEM_GIN)
    message(STATUS "rocshmem GIN backend support enabled")
    add_definitions(-DENABLE_ROCSHMEM_GIN)

    # Auto-detect paths from mono-repo layout
    get_filename_component(_mono_root "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)
    if (NOT ROCSHMEM_INSTALL_DIR)
        if(EXISTS "${_mono_root}/projects/rccl/ext/rocshmem/lib/librocshmem.a")
            set(ROCSHMEM_INSTALL_DIR "${_mono_root}/projects/rccl/ext/rocshmem")
            message(STATUS "rccl-tests: found rocshmem install at ${ROCSHMEM_INSTALL_DIR}")
        else()
            message(FATAL_ERROR "ENABLE_ROCSHMEM_GIN is ON but ROCSHMEM_INSTALL_DIR is not set "
                    "and no install found at ${_mono_root}/projects/rccl/ext/rocshmem")
        endif()
    endif()
    if (NOT RCCL_SOURCE_DIR AND EXISTS "${_mono_root}/projects/rccl/CMakeLists.txt")
        set(RCCL_SOURCE_DIR "${_mono_root}/projects/rccl")
    endif()
    if (NOT ROCSHMEM_SOURCE_DIR AND EXISTS "${_mono_root}/projects/rocshmem/CMakeLists.txt")
        set(ROCSHMEM_SOURCE_DIR "${_mono_root}/projects/rocshmem")
    endif()
    # Installed rocshmem headers (rocshmem_config.h) are needed globally:
    # RCCL's public nccl_device.h → gin_anvil_sdma.h → anvil_device.hpp
    # → log.hpp → rocshmem/rocshmem_config.h.  Must come before source
    # includes so the generated config is found first.
    include_directories(BEFORE ${ROCSHMEM_INSTALL_DIR}/include)
    # Internal rocshmem headers (sdma/anvil_device.hpp) needed globally.
    if (ROCSHMEM_SOURCE_DIR)
        include_directories(${ROCSHMEM_SOURCE_DIR}/src)
    endif()

    find_package(rocshmem REQUIRED CONFIG
        PATHS ${ROCSHMEM_INSTALL_DIR}/lib/cmake/rocshmem NO_DEFAULT_PATH)
    # hsakmtTargets.cmake bakes in RHEL CI paths (/usr/lib64/libc.so,
    # -L/__w/rockrel/...) that don't exist on Ubuntu.  Scrub them and
    # re-add the correct libdrm search path — same fix as rocshmem's
    # own CMakeLists.txt.
    if(TARGET hsakmt::hsakmt)
        get_target_property(_hsakmt_iface_libs hsakmt::hsakmt INTERFACE_LINK_LIBRARIES)
        if(_hsakmt_iface_libs)
            list(FILTER _hsakmt_iface_libs EXCLUDE REGEX "(/libc\\.so|/__w/)")
            find_library(_DRM_LIB drm HINTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
            if(_DRM_LIB)
                get_filename_component(_drm_dir "${_DRM_LIB}" DIRECTORY)
                list(PREPEND _hsakmt_iface_libs "-L${_drm_dir}")
                unset(_drm_dir)
            endif()
            set_target_properties(hsakmt::hsakmt PROPERTIES
                INTERFACE_LINK_LIBRARIES "${_hsakmt_iface_libs}")
            unset(_hsakmt_iface_libs)
        endif()
    endif()
    message(STATUS "rocshmem: roc::rocshmem from ${ROCSHMEM_INSTALL_DIR}")
endif()

set(ROCM_USE_DEV_COMPONENT OFF)  # This repo doesn't have a dev component

# Build the upstream OS abstraction with RCCL's HIP toolchain.
add_library(nccl_tests_options INTERFACE)
target_compile_definitions(nccl_tests_options INTERFACE NCCL_OS_LINUX NCCL_GIN_PROXY_ENABLE=1)
set(NCCL_OS_LINUX ON)
add_subdirectory(os)

# Build the upstream verifiable library with the active CUDA or HIP backend.
add_subdirectory(verifiable)

# Add all of the tests
add_subdirectory(src)

# Host unit tests (GTest) are for standalone/cmake dev builds only. TheRock
# comm-libs configures rccl-tests as a sub-project; find_package(GTest) there
# requires super-project BUILD_DEPS wiring we do not have, so skip test/.
if(BUILD_RCCL_TESTS_HOST_UNIT_TESTS AND NOT DEFINED THEROCK_STAGE_INSTALL_ROOT)
  add_subdirectory(test)
elseif(DEFINED THEROCK_STAGE_INSTALL_ROOT)
  message(STATUS "rccl-tests: skipping host unit tests (TheRock super-project build)")
endif()

rocm_setup_version(VERSION "2.19.6")

# Create ROCm standard packages
rocm_create_package(
    NAME rccl-tests
    DESCRIPTION "Tests for the ROCm Communication Collectives Library"
    MAINTAINER "RCCL Maintainer <rccl-maintainer@amd.com>"
)
