# Copyright (c) 2020 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#Follow "README.md" to generate square.cpp if it's missing

cmake_minimum_required(VERSION 3.21)

# Platform-specific ROCM_PATH setup
if(WIN32)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    elseif(DEFINED ENV{HIP_PATH})
      set(ROCM_PATH $ENV{HIP_PATH} CACHE STRING "ROCM Path from HIP_PATH")
    else()
      # Derive ROCM_PATH from compiler location (more reliable than relative source paths)
      # CMAKE_CXX_COMPILER = .../build/lib/llvm/bin/clang++.exe -> go up 3 levels to get build dir
      get_filename_component(COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
      get_filename_component(ROCM_PATH "${COMPILER_DIR}/../../.." ABSOLUTE)
      set(ROCM_PATH "${ROCM_PATH}" CACHE STRING "ROCM Path (from compiler)")
    endif()
  endif()
  
  # CUDA path for hipify-clang on Windows
  if(NOT DEFINED CUDA_PATH)
    if(DEFINED ENV{CUDA_PATH})
      set(CUDA_PATH $ENV{CUDA_PATH} CACHE STRING "CUDA Path")
    else()
      # Auto-detect CUDA installation
      file(GLOB CUDA_VERSIONS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
      if(CUDA_VERSIONS)
        list(SORT CUDA_VERSIONS)
        list(GET CUDA_VERSIONS -1 CUDA_PATH)
        set(CUDA_PATH "${CUDA_PATH}" CACHE STRING "CUDA Path (auto-detected)")
      else()
        message(FATAL_ERROR "CUDA not found. Please set CUDA_PATH environment variable or -DCUDA_PATH=<path>")
      endif()
    endif()
  endif()
  
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
  
elseif(UNIX)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
  endif()
  # Search for rocm in common locations
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

project(square LANGUAGES CXX HIP)

# Create square.cpp using platform-specific hipify tool
set(SQUARE_CPP "${CMAKE_CURRENT_BINARY_DIR}/square.cpp")

# Allow user to provide a pre-converted file
if(DEFINED HIPIFY_SKIP AND HIPIFY_SKIP)
  if(NOT EXISTS "${SQUARE_CPP}")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/square.cpp")
      file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/square.cpp" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
    else()
      message(FATAL_ERROR "HIPIFY_SKIP is set but no square.cpp found.")
    endif()
  endif()
elseif(WIN32)
  # On Windows, use hipify-clang.exe
  
  # Allow user to override CLANG_RESOURCE_DIR
  if(NOT DEFINED CLANG_RESOURCE_DIR)
    # Find clang resource directory - TheRock layout: ${ROCM_PATH}/lib/llvm/lib/clang/<version>
    file(GLOB CLANG_VERSIONS "${ROCM_PATH}/lib/llvm/lib/clang/*")
    if(CLANG_VERSIONS)
      list(SORT CLANG_VERSIONS)
      list(GET CLANG_VERSIONS -1 CLANG_RESOURCE_DIR)
    else()
      message(FATAL_ERROR "Clang resource directory not found. Please set -DCLANG_RESOURCE_DIR=<path>")
    endif()
  endif()
  
  # Find hipify-clang.exe
  find_program(HIPIFY_CLANG_EXE 
    NAMES hipify-clang.exe hipify-clang
    HINTS 
      "${ROCM_PATH}/bin"
      "$ENV{HIP_PATH}/bin"
  )
  
  if(NOT HIPIFY_CLANG_EXE)
    set(HIPIFY_CLANG_EXE "${ROCM_PATH}/bin/hipify-clang.exe")
  endif()
  
  # Convert all paths to use forward slashes consistently
  file(TO_CMAKE_PATH "${CLANG_RESOURCE_DIR}" CLANG_RESOURCE_DIR_FWD)
  file(TO_CMAKE_PATH "${CUDA_PATH}" CUDA_PATH_FWD)
  file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/square.cu" SOURCE_FILE_FWD)
  file(TO_CMAKE_PATH "${SQUARE_CPP}" OUTPUT_FILE_FWD)
  
  # Execute hipify-clang
  execute_process(
    COMMAND "${HIPIFY_CLANG_EXE}"
      "--clang-resource-directory=${CLANG_RESOURCE_DIR_FWD}"
      "--cuda-path=${CUDA_PATH_FWD}"
      "${SOURCE_FILE_FWD}"
      "-o" "${OUTPUT_FILE_FWD}"
    RESULT_VARIABLE HIPIFY_RESULT
    OUTPUT_VARIABLE HIPIFY_OUTPUT
    ERROR_VARIABLE HIPIFY_ERROR
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  
  if(NOT HIPIFY_RESULT EQUAL 0)
    if(NOT EXISTS "${SQUARE_CPP}")
      message(FATAL_ERROR "hipify-clang failed. Error: ${HIPIFY_ERROR}")
    endif()
  endif()
  
else()
  # On Unix, use hipify-perl
  execute_process(COMMAND sh -c "${ROCM_PATH}/bin/hipify-perl \
           ${CMAKE_CURRENT_SOURCE_DIR}/square.cu > ${CMAKE_CURRENT_BINARY_DIR}/square.cpp")
endif()

# Create the excutable
if(TARGET build_intro)
  set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
  set(EXCLUDE_OPTION)
endif()

# Mark source file as HIP code (contains __global__ kernels)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/square.cpp PROPERTIES LANGUAGE HIP)

add_executable(square ${EXCLUDE_OPTION} ${CMAKE_CURRENT_BINARY_DIR}/square.cpp)

# Set RPATH so executable can find HIP libraries at runtime
if(UNIX)
  set_target_properties(square PROPERTIES
    BUILD_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH "${ROCM_PATH}/lib"
  )
endif()

if(TARGET build_intro)
  add_dependencies(build_intro square)
endif()
