# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.27)
project(clr)

##########
# Defaults
##########
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"
    OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    add_compile_options("/wd4267" "/wd4244" "/wd4996")
    # Enable parallel compilation on MSVC.
    if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
        add_compile_options("/MP$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
    endif()
    string(REPLACE "/GR" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "/W3" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    add_compile_options($<$<CONFIG:Debug>:-D_ITERATOR_DEBUG_LEVEL=0>)
    # Must be set here, before add_subdirectory(rocclr) below, so rocclr and
    # hipamd end up linked against the same MSVC runtime. hipamd/CMakeLists.txt
    # sets this too (for standalone hipamd configs), but by then rocclr has
    # already been configured with CMake's implicit default (/MD), causing a
    # defaultlib 'MSVCRT' conflict and unresolved CRT externals at link time.
    option(__HIP_USE_MSVC_STATIC_RUNTIME "Link MSVC runtime library statically" OFF)
    if(NOT __HIP_USE_MSVC_STATIC_RUNTIME)
        set(__HIP_MSVC_RUNTIME_SUFFIX "DLL")
    endif()
    if(NOT CMAKE_MSVC_RUNTIME_LIBRARY)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>${__HIP_MSVC_RUNTIME_SUFFIX}")
    endif()
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # Unlike when targeting MSFT's MSVC, when targeting clang-cl,
        # CMake does not add these automatically.  Add them ourselves.
        add_compile_options("/DWIN32" "/D_WINDOWS" "/EHsc")
    endif()
endif()

if (AMD_COMPUTE_WIN)
    set(AMD_LIBELF_PATH "${AMD_COMPUTE_WIN}/hsail-compiler/lib/loaders/elf/utils/libelf")
    set(AMD_SC_PATH "${AMD_COMPUTE_WIN}/sc")
    message("Public Compute Windows build path: ${AMD_COMPUTE_WIN}, SC: ${AMD_SC_PATH}, LibElf: ${AMD_LIBELF_PATH}")
endif()

option(CLR_BUILD_HIP "Build HIP" OFF)
option(CLR_BUILD_OCL "Build OCL" OFF)
option(CLR_HRR_CODEGEN_CHECK "Generate and validate HRR generated sources during CLR builds" OFF)
option(USE_NEW_HOSTCALL_IMPL
  "Use the new phase/occupancy-based hostcall implementation" OFF)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/CLRHardening.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/CLRWarnings.cmake")

#############
# Build steps
#############

if(NOT DEFINED ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE STRING "ROCm installation directory.")
  else()
    set(ROCM_PATH "/opt/rocm" CACHE STRING "ROCm installation directory.")
  endif()
endif()
message(STATUS "ROCM_PATH: ${ROCM_PATH}")

if(CLR_BUILD_HIP)
    message(STATUS "HIP Common Directory: ${HIP_COMMON_DIR}")
    if(NOT DEFINED HIP_COMMON_DIR)
        message(FATAL_ERROR "Please pass HIP using -DHIP_COMMON_DIR. HIP_COMMON_DIR is incorrect")
    endif()
    # HIP_PLATFORM selects the HIP backend: "amd" (default) or "nvidia".
    # Pass -DHIP_PLATFORM=nvidia explicitly for NVIDIA builds.
    # Note: ENV{HIP_PLATFORM} (hipcc's env var) is intentionally not read here.
    if(NOT DEFINED HIP_PLATFORM)
        set(HIP_PLATFORM "amd")
        message(STATUS "HIP_PLATFORM defaulting to: ${HIP_PLATFORM}")
    else()
        message(STATUS "HIP_PLATFORM set to: ${HIP_PLATFORM}")
    endif()
endif()

option(BUILD_SHARED_LIBS "Build the shared library" ON)

# Static HIP embeds rocclr objects into amdhip64 and requires BUILD_STATIC_LIBS.
# OpenCL uses BUILD_SHARED_LIBS only to select the amdocl library type.
if(CLR_BUILD_HIP AND NOT BUILD_SHARED_LIBS)
    add_compile_definitions(BUILD_STATIC_LIBS)
endif()

# Applies the host sanitizer flags to this scope, so it has to sit below BUILD_SHARED_LIBS,
# which selects the sanitizer runtime linkage.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/CLRSanitizers.cmake")

# rocclr is only used for the AMD HIP path and OpenCL; the NVIDIA path does not use it.
if((CLR_BUILD_HIP AND HIP_PLATFORM STREQUAL "amd") OR CLR_BUILD_OCL)
    add_subdirectory(rocclr)
elseif(NOT CLR_BUILD_HIP AND NOT CLR_BUILD_OCL)
    message(FATAL_ERROR "Please enable building of one or more of the below runtimes:\n- HIP (-DCLR_BUILD_HIP=ON)\n- OpenCL (-DCLR_BUILD_OCL=ON)")
endif()
if(CLR_BUILD_HIP)
    add_subdirectory(hipamd)
endif()
if(CLR_BUILD_OCL)
    add_subdirectory(opencl)
endif()

#############################
# Code formatting
#############################
# Target: clangformat
find_program(CLANGFORMAT_EXE clang-format PATHS "${ROCM_PATH}/lib/llvm/bin")
if(CLANGFORMAT_EXE)
    file(GLOB_RECURSE FORMAT_SOURCE_FILE_LIST *.cpp *.hpp *.hh *.h *.cc *.c)
    add_custom_target(clangformat COMMAND ${CLANGFORMAT_EXE} -style=file -i ${FORMAT_SOURCE_FILE_LIST}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()
