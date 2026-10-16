################################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.10)
# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT WIN32)
  if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
  endif()
endif()

project(rocjpegnegativetest)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

# rocJPEG sample build type
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "rocJPEG Default Build Type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
if(NOT WIN32)
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(ROCJPEG_SAMPLE_COMPILE_OPTIONS -O0 -gdwarf-4 -Wall)
  else()
    set(ROCJPEG_SAMPLE_COMPILE_OPTIONS -O3 -DNDEBUG -fPIC -Wall)
  endif()
endif()

set (HIP_PLATFORM amd CACHE STRING "HIP platform")
find_package(HIP QUIET)
find_package(rocjpeg QUIET)
if(NOT WIN32)
    find_package(rocprofiler-register QUIET)
endif()

if(HIP_FOUND AND rocjpeg_FOUND AND (WIN32 OR rocprofiler-register_FOUND))
    # HIP
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)
    # rocJPEG and utils
    include_directories (${rocjpeg_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocjpeg::rocjpeg)
    # rocprofiler-register
    if(rocprofiler-register_FOUND)
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocprofiler-register::rocprofiler-register)
    endif()
    # sample app exe
    list(APPEND SOURCES ${PROJECT_SOURCE_DIR} rocjpegnegativetest.cpp rocjpeg_api_negative_tests.cpp)
    add_executable(${PROJECT_NAME} ${SOURCES})
    target_compile_options(${PROJECT_NAME} PRIVATE ${ROCJPEG_SAMPLE_COMPILE_OPTIONS})
    if(WIN32)
        target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    endif()
    target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})
else()
    message("-- ERROR!: ${PROJECT_NAME} excluded! unmet dependencies")
    if (NOT HIP_FOUND)
        message(FATAL_ERROR "-- ERROR!: HIP Not Found!")
    endif()
    if (NOT rocjpeg_FOUND)
        message(FATAL_ERROR "-- ERROR!: rocJPEG Not Found!")
    endif()
    if (NOT WIN32 AND NOT rocprofiler-register_FOUND)
        message(FATAL_ERROR "-- ERROR!: rocprofiler-register Not Found!")
    endif()
endif()
