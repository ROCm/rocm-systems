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

cmake_minimum_required(VERSION 3.21)

if(UNIX)
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

project(hipDispatchLatency LANGUAGES CXX HIP)

set(CMAKE_BUILD_TYPE Release)

# Detect GPU architectures and setup OFFLOAD_ARCH_FLAGS
include(${CMAKE_CURRENT_SOURCE_DIR}/../../common/DetectGPUArchs.cmake)

# Create the excutable
if(TARGET build_utils)
  set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
  set(EXCLUDE_OPTION)
endif()

# Mark source files as HIP code
set_source_files_properties(hipDispatchLatency.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(hipDispatchEnqueueRateMT.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(test_kernel.cpp PROPERTIES LANGUAGE HIP)

add_executable(hipDispatchLatency ${EXCLUDE_OPTION} hipDispatchLatency.cpp)
add_executable(hipDispatchEnqueueRateMT ${EXCLUDE_OPTION} hipDispatchEnqueueRateMT.cpp)

# Generate code object using CMake's HIP language support
# Note: Build for all detected architectures to support multiple GPUs (like hipcc does)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_kernel.code
  COMMAND ${CMAKE_HIP_COMPILER} -x hip --cuda-device-only ${OFFLOAD_ARCH_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/test_kernel.cpp -o ${CMAKE_CURRENT_BINARY_DIR}/test_kernel.code
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_kernel.cpp
  COMMENT "Generating code object test_kernel.code for ${CMAKE_HIP_ARCHITECTURES}"
  VERBATIM
)

add_custom_target(codeobj2 ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/test_kernel.code)

add_dependencies(hipDispatchLatency codeobj2)
add_dependencies(hipDispatchEnqueueRateMT codeobj2)

target_include_directories(hipDispatchLatency PRIVATE ../../common)
target_include_directories(hipDispatchEnqueueRateMT PRIVATE ../../common)

if(UNIX)
  target_link_libraries(hipDispatchEnqueueRateMT pthread)
  # Set RPATH so executables can find HIP libraries at runtime
  set_target_properties(hipDispatchLatency hipDispatchEnqueueRateMT PROPERTIES
    BUILD_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH "${ROCM_PATH}/lib"
  )
endif()

set_property(TARGET hipDispatchLatency PROPERTY CXX_STANDARD 11)
set_property(TARGET hipDispatchEnqueueRateMT PROPERTY CXX_STANDARD 11)

if(TARGET build_utils)
add_dependencies(build_utils hipDispatchLatency hipDispatchEnqueueRateMT)
endif()