################################################################################
# ROCm Systems CMAKE Configuration
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
################################################################################

cmake_minimum_required(VERSION 3.18.0)

project(rocm-systems 
    VERSION 7.1.0
    DESCRIPTION "ROCm Systems CMAKE Config"
    LANGUAGES C CXX)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build: None Debug Release RelWithDebInfo."
        FORCE)
endif()

# Include GNUInstallDirs for standard installation paths
include(GNUInstallDirs)

# Set default ROCM_PATH if not defined
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory")
endif()

# Print build configuration
message(STATUS "")
message(STATUS "================================")
message(STATUS "ROCm Systems Build Configuration")
message(STATUS "================================")
message(STATUS "CMake Version:        ${CMAKE_VERSION}")
message(STATUS "Build Type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "ROCm Path:            ${ROCM_PATH}")
message(STATUS "Install Prefix:       ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Source Directory:     ${CMAKE_SOURCE_DIR}")
message(STATUS "Binary Directory:     ${CMAKE_BINARY_DIR}")
message(STATUS "")

# Add help target to show available options
add_custom_target(rocm-systems-help
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Configuration Options:"
    COMMAND ${CMAKE_COMMAND} -E echo "  -DROCM_PATH=/path/to/rocm           - Set ROCm installation path"
    COMMAND ${CMAKE_COMMAND} -E echo "  -DCMAKE_INSTALL_PREFIX=/install/dir - Set installation directory"
    COMMAND ${CMAKE_COMMAND} -E echo "  -DCMAKE_BUILD_TYPE=Release|Debug    - Set build type"
    COMMENT "Displaying ROCm Systems build help")

################################################################################
# Package Configuration for find_package() Support
################################################################################

# Include CMake package configuration helpers
include(CMakePackageConfigHelpers)

# Set package installation directory
set(ROCM_SYSTEMS_CONFIG_INSTALL_DIR 
    "${CMAKE_INSTALL_LIBDIR}/cmake/rocm-systems" 
    CACHE STRING "Installation directory for CMake package config files")

# Create a list of available components based on what was built
set(ROCM_SYSTEMS_AVAILABLE_COMPONENTS "")


list(APPEND ROCM_SYSTEMS_AVAILABLE_COMPONENTS "rdc")
list(APPEND ROCM_SYSTEMS_AVAILABLE_COMPONENTS "rocm-core")
list(APPEND ROCM_SYSTEMS_AVAILABLE_COMPONENTS "rocm_smi")
list(APPEND ROCM_SYSTEMS_AVAILABLE_COMPONENTS "rocprofiler-register")
list(APPEND ROCM_SYSTEMS_AVAILABLE_COMPONENTS "rocprofiler-sdk")
list(APPEND ROCM_SYSTEMS_AVAILABLE_COMPONENTS "rocprofiler-systems")
list(APPEND ROCM_SYSTEMS_AVAILABLE_COMPONENTS "rocr-runtime")
list(APPEND ROCM_SYSTEMS_AVAILABLE_COMPONENTS "hip")

# Configure the package config file template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocm-systems-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rocm-systems-config.cmake"
    INSTALL_DESTINATION "${ROCM_SYSTEMS_CONFIG_INSTALL_DIR}"
    PATH_VARS CMAKE_INSTALL_PREFIX CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_BINDIR
)

# Generate the package version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rocm-systems-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install the package configuration files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/rocm-systems-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rocm-systems-config-version.cmake"
    DESTINATION "${ROCM_SYSTEMS_CONFIG_INSTALL_DIR}"
)

# Export targets if any exist (individual projects manage their own exports)
# This creates a master find script that can locate all sub-projects

message(STATUS "")
message(STATUS "Package Configuration:")
message(STATUS "  Available Components: ${ROCM_SYSTEMS_AVAILABLE_COMPONENTS}")
message(STATUS "  Config Install Dir:   ${ROCM_SYSTEMS_CONFIG_INSTALL_DIR}")
message(STATUS "")


