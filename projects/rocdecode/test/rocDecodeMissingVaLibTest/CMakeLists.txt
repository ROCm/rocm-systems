################################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

# Windows-only: checks that rocdecode.dll fails cleanly when the TheRock VA-API
# DLLs cannot be found. See rocdecodemissingvalibtest.cpp.
cmake_minimum_required(VERSION 3.10)
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)

project(rocdecodemissingvalibtest)

if(NOT WIN32)
  message(FATAL_ERROR "-- ERROR!: ${PROJECT_NAME} is Windows-only")
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "rocdecode Default Build Type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

set(HIP_PLATFORM amd CACHE STRING "HIP platform")
find_package(HIP QUIET)
find_package(rocdecode QUIET)

if(HIP_FOUND AND rocdecode_FOUND)
  add_executable(${PROJECT_NAME} rocdecodemissingvalibtest.cpp)
  target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
  # rocdecode.h includes hip/hip_runtime.h. The test loads a relocated copy of
  # rocdecode.dll at runtime, so it must not link rocdecode itself; it only
  # needs the headers and the installed DLL's location.
  target_include_directories(${PROJECT_NAME} PRIVATE ${rocdecode_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} hip::host)
  target_compile_definitions(${PROJECT_NAME} PRIVATE
    ROCDECODE_DLL_PATH=L"$<TARGET_FILE:rocdecode::rocdecode>")
else()
  message("-- ERROR!: ${PROJECT_NAME} excluded! unmet dependencies")
  if(NOT HIP_FOUND)
    message(FATAL_ERROR "-- ERROR!: HIP Not Found!")
  endif()
  if(NOT rocdecode_FOUND)
    message(FATAL_ERROR "-- ERROR!: rocdecode Not Found!")
  endif()
endif()
