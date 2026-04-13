# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Top-level CMake file for AIS

# CMake version according to latest ROCm platform requirements
cmake_minimum_required(VERSION 3.21)

# Set the default CMake build type
#
# NOTE: This has to be done BEFORE the project is declared!
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type")
endif()

# Set the library version in a variable so we can use
# it in *.in files and avoid version mismatch.
set(AIS_LIBRARY_MAJOR 0)
set(AIS_LIBRARY_MINOR 2)
set(AIS_LIBRARY_PATCH 0)
set(AIS_LIBRARY_VERSION "${AIS_LIBRARY_MAJOR}.${AIS_LIBRARY_MINOR}.${AIS_LIBRARY_PATCH}")

# Set the SOVERSION to the major library version (this is the default
# but we set it explicitly to convey intent).
#
# Changes to minor and patch releases must NEVER break the ABI. This
# includes things like reordering struct fields that don't break the API.
set(AIS_LIBRARY_SOVERSION "${AIS_LIBRARY_MAJOR}")

project("hipfile"
    VERSION ${AIS_LIBRARY_VERSION}
    DESCRIPTION "Direct-to-GPU IO for AMD's ROCm platform"
    LANGUAGES C CXX HIP
)

# Switch between building static and shared libraries
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# Set the path to the include directories for later use
# Do this early so we can use these later, in functions, etc.
set(HIPFILE_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(HIPFILE_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(HIPFILE_AMD_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/amd_detail")
set(HIPFILE_NVIDIA_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/nvidia_detail")
set(HIPFILE_SRC_COMMON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/common")
set(HIPFILE_TEST_COMMON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/test/common")
set(HIPFILE_AMD_TEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/test/amd_detail")

# Set the list of build types for ccmake/CMake GUI
#
# This turns the build type box into a multi-select so you don't have to
# enter the names manually
#
# "None" sets nothing, so it's easier to override -O flags, etc.
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel" "None")

# Enable CTest (the include line automatically calls enable_testing())
include(CTest)

# Enable options with dependencies
include(CMakeDependentOption)

# Ensure compile_commands.json has implicit header paths
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

#-----------------------------------------------------------------------------
# Get the HIP platform
#-----------------------------------------------------------------------------
# The default is "amd" if CMAKE_HIP_PLATFORM hasn't been set
if(NOT CMAKE_HIP_PLATFORM)
    set(CMAKE_HIP_PLATFORM "amd" CACHE STRING "HIP platform to build with")
endif()
set_property(CACHE CMAKE_HIP_PLATFORM PROPERTY STRINGS "amd" "nvidia")

# Set the HIP_PLATFORM environment variable so that HIP headers are
# parsed correctly
set(ENV{HIP_PLATFORM} ${CMAKE_HIP_PLATFORM})

# Make sure the HIP platform is valid
if(NOT CMAKE_HIP_PLATFORM STREQUAL "amd" AND NOT CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    message(FATAL_ERROR "Invalid CMAKE_HIP_PLATFORM: '${CMAKE_HIP_PLATFORM}'. Allowed values are 'amd' or 'nvidia'.")
endif()

# Set the platform variables (internal only)
if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    set(AIS_BUILD_AMD_DETAIL OFF)
    set(AIS_BUILD_NVIDIA_DETAIL ON)
else()
    set(AIS_BUILD_AMD_DETAIL ON)
    set(AIS_BUILD_NVIDIA_DETAIL OFF)
endif()

#-----------------------------------------------------------------------------
# Include our custom CMake code in the module path
#-----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#-----------------------------------------------------------------------------
# Set ROCm Version and Install Path
#
# Values are set with the following prioritization in case of redundant declarations:
# 1) CMake Argument
# 2) Environment Variables
# 3) Default Values
#
# Note: project(... LANGUAGES HIP) only looks at $ENV{ROCM_PATH} and will ignore
#       ROCM_PATH set as a CMake Argument.
#-----------------------------------------------------------------------------
# If ROCM_VERSION was set in the environment, seed it with that
if(NOT DEFINED ROCM_VERSION AND DEFINED ENV{ROCM_VERSION})
    set(ROCM_VERSION "$ENV{ROCM_VERSION}")
endif()

# If ROCM_PATH was set in the environment, seed it with that
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_INITIAL_PATH "$ENV{ROCM_PATH}")
elseif(DEFINED ROCM_VERSION)
    # Infer default based on ROCM_VERSION
    set(ROCM_INITIAL_PATH "/opt/rocm-${ROCM_VERSION}")
else()
    # Default install location symlink
    set(ROCM_INITIAL_PATH "/opt/rocm")
endif()
set(ROCM_PATH "${ROCM_INITIAL_PATH}" CACHE PATH "The path to the ROCm installation")

# Fallback if ROCM_VERSION not specified. Get from version file in ROCM_PATH
if(NOT DEFINED ROCM_VERSION)
    file(READ "${ROCM_PATH}/.info/version" _rocm_version_file)
    string(REGEX MATCH "^([0-9]+\\.[0-9]+\\.[0-9]+)" _matched_rocm_version "${_rocm_version_file}")
    set(ROCM_VERSION "${_matched_rocm_version}")
endif()
set(ROCM_VERSION "${ROCM_VERSION}" CACHE STRING "The version of ROCm to build with")

message(STATUS "Using ROCM_VERSION: ${ROCM_VERSION}")
message(STATUS "ROCM_PATH set to: ${ROCM_PATH}")

# Add ROCm search paths
list(APPEND CMAKE_PREFIX_PATH
    ${ROCM_PATH}/llvm
    ${ROCM_PATH}
    ${ROCM_PATH}/hip
    /opt/rocm/llvm
    /opt/rocm
    /opt/rocm/hip
)

# Set hipFile Install Path to the ROCm directory
# Note: CMAKE_INSTALL_PREFIX is set to a default by project().
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "The path where hipFile should be installed" FORCE)
endif()

# Fix library install directory to "lib" to be inline with the rest of ROCm
# Note: If testing is enabled, installing GTest calls GNUInstallDirs which will
#       set this before ROCMInstallTargets can set it.
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Directory name for installed ROCm libraries")

#-----------------------------------------------------------------------------
# Import useful things for later
#-----------------------------------------------------------------------------
include(AISAddLibraries)

#-----------------------------------------------------------------------------
# Options
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Code coverage (via llvm-cov)
#
# To use with the hipFile library:
#   - Build with AIS_USE_CODE_COVERAGE set to ON
#   - Run `ctest .`
#
#   This should create a default.profraw file in each test directory
#
# In each test directory:
#   - Run `llvm-profdata merge -output=<hipfile, etc.>.profdata default.profraw`
#
# The show command gives line-by-line coverage, report gives a summary
#   - Run `llvm-cov <show|report> -instr-profile=<hipfile, etc.>.profdata`
#
# You can export the data to lcov or json
#   - Run `llvm-cov export -instr-profile=ais.profdata -format=<json|lcov> > coverage.<lcov|json>`
#-----------------------------------------------------------------------------
option(AIS_USE_CODE_COVERAGE "Build with code coverage flags (llvm C++ only)" OFF)

#----------------------------------------------
# Clang "safe buffer" warnings and suggestions
#----------------------------------------------
include(AISClangSafeBuffers)

#----------------
# Run clang-tidy
#----------------
include(AISClangTidy)

#---------------------------------
# Run include-what-you-use (IWYU)
#---------------------------------
include(AISIWYU)

#-------------------
# Sanitizer options
#-------------------
include(AISSanitizers)

#------------------
# Example programs
#------------------
option(AIS_INSTALL_EXAMPLES "Install example programs" ON)

#------------------
# Tool programs
#------------------
option(AIS_INSTALL_TOOLS "Install tool programs" ON)


#------
# docs
#------
include(AISDocumentation)

#-----------------------------------------------------------------------------
# Find important packages for building the software
#-----------------------------------------------------------------------------

find_package(hsa-runtime64 CONFIG REQUIRED)
if(${hsa-runtime64_FOUND})
    message(STATUS "hsa-runtime64 found @  ${hsa-runtime64_DIR} ")
else()
    message(NOTICE "find_package did NOT find hsa-runtime64")
endif()

find_package(hip CONFIG REQUIRED)
if(${hip_FOUND})
    message(STATUS "hip found @  ${hip_DIR} ")
else()
    message(NOTICE "find_package did NOT find hip")
endif()

# Find the cuFile library and headers on NVIDIA
if(AIS_BUILD_NVIDIA_DETAIL)
    find_package(CUDAToolkit REQUIRED)

    if(CMAKE_VERSION VERSION_LESS "3.28")
        enable_language(CUDA)
        # Set a default architecture if not provided.
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 80)
        endif()
    endif()

    find_library(CUFILE_LIBRARY cufile PATHS ${CUDAToolkit_LIBRARY_DIR})

    if(NOT CUFILE_LIBRARY)
        message(FATAL_ERROR "cuFile library not found")
    endif()
endif()

if(NOT AIS_CAPABLE_DIR)
    set(AIS_CAPABLE_DIR "/tmp" CACHE STRING "Directory to use for e2e tests.")
endif()

#-----------------------------------------------------------------------------
# Configure hipFile
#-----------------------------------------------------------------------------

# Add the src directory depending on the HIP platform
if(AIS_BUILD_NVIDIA_DETAIL)
    add_subdirectory(src/nvidia_detail)
else()
    add_subdirectory(src/amd_detail)
endif()

# Add the testing directories
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
if(BUILD_TESTING AND AIS_BUILD_AMD_DETAIL)
    add_subdirectory(test/amd_detail)
endif()

# Add the example directory
if(AIS_INSTALL_EXAMPLES)
    add_subdirectory(examples/aiscp)
endif()

# Add tools directory
if(AIS_INSTALL_TOOLS AND AIS_BUILD_AMD_DETAIL)
    add_subdirectory(tools/ais-stats)
endif()


#-----------------------------------------------------------------------------
# Set up installs (must come AFTER target creation)
#-----------------------------------------------------------------------------
include(AISUseROCmCMake)
include(AISInstall)
