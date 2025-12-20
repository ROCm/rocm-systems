################################################################################
# FINDROCm CMAKE Configuration
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
################################################################################

cmake_minimum_required(VERSION 3.18.0)

project(FINDROCm 
    VERSION 7.1.0
    DESCRIPTION "FINDROCm CMAKE Config")

# Include GNUInstallDirs for standard installation paths
include(GNUInstallDirs)

# Set default ROCM_PATH if not defined
if(NOT DEFINED ROCM_PATH)
  message(STATUS "ROCM_PATH NOT SET")
endif()

# Print build configuration
message(STATUS "")
message(STATUS "================================")
message(STATUS "FINDROCm CMAKE Config")
message(STATUS "================================")
message(STATUS "CMake Version:        ${CMAKE_VERSION}")
message(STATUS "ROCm Path:            ${ROCM_PATH}")
message(STATUS "Install Prefix:       ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Source Directory:     ${CMAKE_SOURCE_DIR}")
message(STATUS "Binary Directory:     ${CMAKE_BINARY_DIR}")
message(STATUS "")

################################################################################
# Package Configuration for find_package() Support
################################################################################

# Include CMake package configuration helpers
include(CMakePackageConfigHelpers)

# Set package installation directory
set(FINDROCM_CONFIG_INSTALL_DIR 
	"${CMAKE_INSTALL_LIBDIR}/cmake/FINDROCm" 
    CACHE STRING "Installation directory for CMake package config files")

# Create a list of available components based on what was built
set(FINDROCM_AVAILABLE_COMPONENTS "")

list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rdc")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocm-core")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocm_smi")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocprofiler-register")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocprofiler-sdk")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocprofiler-sdk-rocpd")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocprofiler-sdk-roctx")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocprofiler-systems")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocr-runtime")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "amd_smi")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hsakmt")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hip")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hiprtc")
#list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hip-lang")

# Math Libs
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "composable_kernel")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hipblaslt")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hipblas-common")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hipblas")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocprim")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hipcub")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hipsolver")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hiprand")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hipfort")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hipfft")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hipsparse")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hipsparselt")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "hiptensor")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocblas")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rccl")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocfft")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocalution")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocthrust")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocsparse")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocsolver")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocrand")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "miopen")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "migraphx")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rpp")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rocroller")
#list(APPEND FINDROCM_AVAILABLE_COMPONENTS "origami")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "rvs")

# LLVM Repo
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "amd_comgr")
list(APPEND FINDROCM_AVAILABLE_COMPONENTS "AMDDeviceLibs")

list(APPEND FINDROCM_AVAILABLE_COMPONENTS "amd-dbgapi")

# Configure the package config file template
configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/FINDROCmConfig.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/FINDROCmConfig.cmake"
    INSTALL_DESTINATION "${FINDROCM_CONFIG_INSTALL_DIR}"
    PATH_VARS CMAKE_INSTALL_PREFIX CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_BINDIR
)

# Generate the package version file
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/FINDROCmConfig-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install the package configuration files
install(FILES
	"${CMAKE_CURRENT_BINARY_DIR}/FINDROCmConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/FINDROCmConfig-version.cmake"
    DESTINATION "${FINDROCM_CONFIG_INSTALL_DIR}"
)

# Export targets if any exist (individual projects manage their own exports)
# This creates a master find script that can locate all sub-projects

message(STATUS "")
message(STATUS "Package Configuration:")
message(STATUS "  Available Components: ${FINDROCM_AVAILABLE_COMPONENTS}")
message(STATUS "  Config Install Dir:   ${FINDROCM_CONFIG_INSTALL_DIR}")
message(STATUS "")


