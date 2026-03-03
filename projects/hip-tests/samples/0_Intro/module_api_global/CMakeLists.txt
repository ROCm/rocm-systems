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

project(module_api_global LANGUAGES CXX HIP)

# Detect GPU architectures and setup OFFLOAD_ARCH_FLAGS
include(${CMAKE_CURRENT_SOURCE_DIR}/../../common/DetectGPUArchs.cmake)

# Create the excutable
if(TARGET build_intro)
set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
set(EXCLUDE_OPTION)
endif()

# Mark source files as HIP (use HIP runtime APIs)
set_source_files_properties(runKernel.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(vcpy_kernel.cpp PROPERTIES LANGUAGE HIP)

add_executable(runKernel1.hip.out ${EXCLUDE_OPTION} runKernel.cpp)

# Generate code object using CMake's HIP language support
# Note: Build for all detected architectures to support multiple GPUs (like hipcc does)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vcpy_kernel.code
  COMMAND ${CMAKE_HIP_COMPILER} -x hip --cuda-device-only ${OFFLOAD_ARCH_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/vcpy_kernel.cpp -o ${CMAKE_CURRENT_BINARY_DIR}/vcpy_kernel.code
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vcpy_kernel.cpp
  COMMENT "Generating code object vcpy_kernel.code for ${CMAKE_HIP_ARCHITECTURES}"
  VERBATIM
)

add_custom_target(codeobj1 ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vcpy_kernel.code)

add_dependencies(runKernel1.hip.out codeobj1)

target_include_directories(runKernel1.hip.out PRIVATE ../../common)

# Set RPATH so executable can find HIP libraries at runtime
if(UNIX)
  set_target_properties(runKernel1.hip.out PROPERTIES
    BUILD_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH "${ROCM_PATH}/lib"
  )
endif()

if(TARGET build_intro)
add_dependencies(build_intro runKernel1.hip.out)
endif()
