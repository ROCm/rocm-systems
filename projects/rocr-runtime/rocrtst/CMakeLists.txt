#
# Top-level CMakeLists.txt for ROCR Test Suite (rocrtst)
# This builds rocrtst as a standalone project after ROCR is installed
#
# Usage:
#   1. Build and install ROCR first
#   2. Use build_all.bat (recommended) OR configure this CMakeLists.txt manually
#   3. For manual build: cmake -DHSA_LIBRARY=<path_to_hsa-runtime64.lib> ...
#

cmake_minimum_required(VERSION 3.5.0)

project(rocrtst)

if(WIN32)
    message(STATUS "ROCR Test Suite (rocrtst) - Standalone build for Windows")

    # Verify that HSA_LIBRARY is provided and exists (required for Windows)
    if(NOT DEFINED HSA_LIBRARY)
        message(FATAL_ERROR "HSA_LIBRARY must be defined for rocrtst build. Use build_all.bat or set manually.")
    endif()

    if(NOT EXISTS "${HSA_LIBRARY}")
        message(FATAL_ERROR "HSA_LIBRARY does not exist: ${HSA_LIBRARY}")
    endif()

    message(STATUS "Using HSA library: ${HSA_LIBRARY}")

    # Set up build type if not specified
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
    endif()

    # Set up rocrtst build variables
    set(ROCRTST_BLD_TYPE "${CMAKE_BUILD_TYPE}")
    set(ROCRTST_BLD_BITS "64")

    # Add subdirectories for the different components
    # Each subdirectory's CMakeLists.txt will handle its own dependencies

    # Build kernels (if LLVM_DIR is available)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Kernels/CMakeLists.txt")
        add_subdirectory(Kernels)
    endif()

    # Build test common utilities (includes gtest internally, don't add gtest separately)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/suites/test_common/CMakeLists.txt")
        add_subdirectory(suites/test_common)
    endif()

    # Build samples
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/samples/CMakeLists.txt")
        add_subdirectory(samples)
    endif()
    message(STATUS "ROCR Test Suite (rocrtst) configuration complete")
endif()