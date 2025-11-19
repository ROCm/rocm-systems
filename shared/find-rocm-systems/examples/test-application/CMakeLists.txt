################################################################################
# Example Test Application Using ROCm Systems
# 
# This example demonstrates how to use find_package(rocm-systems) to build
# a test application that depends on ROCm components.
#
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
################################################################################

cmake_minimum_required(VERSION 3.18)
project(ROCmSystemsTestApp VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

################################################################################
# Find ROCm Systems Package
################################################################################

# Option 1: Find with specific components (recommended)
find_package(rocm-systems REQUIRED COMPONENTS
    rocm-core
    rocr-runtime
    rocprofiler-sdk
)

################################################################################
# Display Found Information
################################################################################

message(STATUS "========================================")
message(STATUS "ROCm Systems Test Application")
message(STATUS "========================================")
message(STATUS "ROCm Systems Version: ${ROCM_SYSTEMS_VERSION}")
message(STATUS "Installation Prefix:  ${ROCM_SYSTEMS_PREFIX}")
message(STATUS "Library Directory:    ${ROCM_SYSTEMS_LIB_DIR}")
message(STATUS "Include Directory:    ${ROCM_SYSTEMS_INCLUDE_DIR}")
message(STATUS "Binary Directory:     ${ROCM_SYSTEMS_BIN_DIR}")
message(STATUS "")
message(STATUS "Available Components: ${ROCM_SYSTEMS_AVAILABLE_COMPONENTS}")
message(STATUS "Found Components:     ${rocm-systems_FOUND_COMPONENTS}")
message(STATUS "")
message(STATUS "Component Status:")
message(STATUS "  rocm-core:          ${rocm-systems_rocm-core_FOUND}")
message(STATUS "  rocr-runtime:       ${rocm-systems_rocr-runtime_FOUND}")
message(STATUS "  rocprofiler-sdk:    ${rocm-systems_rocprofiler-sdk_FOUND}")
message(STATUS "========================================")

################################################################################
# Example 1: Basic ROCm Info Tool
################################################################################

add_executable(rocm_info_test rocm_info_test.cpp)

target_include_directories(rocm_info_test PRIVATE 
    ${ROCM_SYSTEMS_INCLUDE_DIRS}
)

# Link against rocm-core if available
if(TARGET rocm-core)
    target_link_libraries(rocm_info_test PRIVATE rocm-core)
endif()

################################################################################
# Example 2: HSA Runtime Test
################################################################################

if(rocm-systems_rocr-runtime_FOUND)
    add_executable(hsa_runtime_test hsa_runtime_test.cpp)
    
    target_include_directories(hsa_runtime_test PRIVATE 
        ${ROCM_SYSTEMS_INCLUDE_DIRS}
    )
    
    # Link against HSA runtime
    # The component may provide hsa-runtime64::hsa-runtime64 imported target
    if(TARGET hsa-runtime64::hsa-runtime64)
        target_link_libraries(hsa_runtime_test PRIVATE hsa-runtime64::hsa-runtime64)
    else()
        # Fallback to finding the library manually
        find_library(HSA_RUNTIME_LIB hsa-runtime64
            PATHS ${ROCM_SYSTEMS_LIB_DIR}
            NO_DEFAULT_PATH
        )
        if(HSA_RUNTIME_LIB)
            target_link_libraries(hsa_runtime_test PRIVATE ${HSA_RUNTIME_LIB})
        endif()
    endif()
    
    message(STATUS "Building hsa_runtime_test")
else()
    message(STATUS "Skipping hsa_runtime_test (rocr-runtime not found)")
endif()

################################################################################
# Summary
################################################################################

message(STATUS "")
message(STATUS "Configured test applications. To build:")
message(STATUS "  cmake --build .")
message(STATUS "")
message(STATUS "To run tests after building:")
message(STATUS "  ./rocm_info_test")
if(TARGET hsa_runtime_test)
    message(STATUS "  ./hsa_runtime_test")
endif()
message(STATUS "")


