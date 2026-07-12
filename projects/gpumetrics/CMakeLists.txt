# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)
project(gpumetrics LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Where to find ROCm (amd_smi, rocprofiler-sdk, hsa).
set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm install prefix")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${ROCM_PATH}/lib/cmake")

option(GPUMETRICS_BUILD_AMDSMI_PLUGIN "Build the amdsmi plugin" ON)
option(GPUMETRICS_BUILD_ROCPROFILER_PLUGIN "Build the rocprofiler plugin" ON)
option(GPUMETRICS_BUILD_CLI "Build the gpumetrics CLI" ON)
option(GPUMETRICS_BUILD_TESTS "Build tests" ON)

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# Core library (no ROCm deps, pure orchestration + dlopen).
add_subdirectory(src)

if(GPUMETRICS_BUILD_AMDSMI_PLUGIN)
  add_subdirectory(plugins/amdsmi)
endif()
if(GPUMETRICS_BUILD_ROCPROFILER_PLUGIN)
  add_subdirectory(plugins/rocprofiler)
endif()
if(GPUMETRICS_BUILD_CLI)
  add_subdirectory(cli)
endif()
if(GPUMETRICS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
