################################################################################
# Example Test Application Using FINDROCm
# 
# This example demonstrates how to use find_package(FINDROCm) to build
# a test application that depends on ROCm components.
#
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
################################################################################

cmake_minimum_required(VERSION 3.18)
project(FINDROCmTestApp VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

################################################################################
# Find FINDROCm Package
################################################################################

# Option 1: Find with specific components (recommended)
find_package(FINDROCm REQUIRED PATHS /opt/AMD COMPONENTS
    rocm-core
    rocr-runtime
    rocprofiler-sdk
)

################################################################################
# Display Found Information
################################################################################

message(STATUS "========================================")
message(STATUS "FINDROCm Test Application")
message(STATUS "========================================")
message(STATUS "FINDROCm Version:     ${FINDROCM_VERSION}")
message(STATUS "Installation Prefix:  ${FINDROCM_PREFIX}")
message(STATUS "Library Directory:    ${FINDROCM_LIB_DIR}")
message(STATUS "Include Directory:    ${FINDROCM_INCLUDE_DIR}")
message(STATUS "Binary Directory:     ${FINDROCM_BIN_DIR}")
message(STATUS "")
message(STATUS "Available Components: ${FINDROCM_AVAILABLE_COMPONENTS}")
message(STATUS "Found Components:     ${FINDROCm_FOUND_COMPONENTS}")
message(STATUS "")
message(STATUS "Component Status:")
message(STATUS "  rocm-core:          ${FINDROCm_rocm-core_FOUND}")
message(STATUS "  rocr-runtime:       ${FINDROCm_rocr-runtime_FOUND}")
message(STATUS "  rocprofiler-sdk:    ${FINDROCm_rocprofiler-sdk_FOUND}")
message(STATUS "========================================")

################################################################################
# Example 1: Basic ROCm Info Tool
################################################################################

add_executable(rocm_info_test rocm_info_test.cpp)

target_include_directories(rocm_info_test PRIVATE 
    ${FINDROCM_INCLUDE_DIRS}
)

# Link against rocm-core if available
if(TARGET rocm-core)
    target_link_libraries(rocm_info_test PRIVATE rocm-core)
endif()

################################################################################
# Example 2: HSA Runtime Test
################################################################################

if(FINDROCm_rocr-runtime_FOUND)
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
	    PATHS ${FINDROCM_LIB_DIR}
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


