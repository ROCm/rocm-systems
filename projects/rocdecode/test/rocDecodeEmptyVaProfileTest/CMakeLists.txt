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

# Linux-only: checks that rocDecGetDecoderCaps fails, rather than reporting every codec as unsupported, when the VA
# driver loads but exposes no decode profile. See rocdecodeemptyvaprofiletest.cpp.
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

project(rocdecodeemptyvaprofiletest)

if(WIN32)
  message(FATAL_ERROR "-- ERROR!: ${PROJECT_NAME} is Linux-only")
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
# The stub VA driver is prebuilt by the rocdecode build and installed next to this file without its .so suffix.
set(STUB_DRIVER_NAME rocdecstub)
set(STUB_DRIVER_PREBUILT ${CMAKE_CURRENT_SOURCE_DIR}/${STUB_DRIVER_NAME}_drv_video.bin)

if(HIP_FOUND AND rocdecode_FOUND AND EXISTS ${STUB_DRIVER_PREBUILT})
  # libva loads <LIBVA_DRIVER_NAME>_drv_video.so from LIBVA_DRIVERS_PATH.
  set(STUB_DRIVER_DIR ${CMAKE_CURRENT_BINARY_DIR}/stub_driver)
  configure_file(${STUB_DRIVER_PREBUILT} ${STUB_DRIVER_DIR}/${STUB_DRIVER_NAME}_drv_video.so COPYONLY)

  add_executable(${PROJECT_NAME} rocdecodeemptyvaprofiletest.cpp)
  target_link_libraries(${PROJECT_NAME} rocdecode::rocdecode hip::host ${CMAKE_DL_LIBS})
  target_compile_definitions(${PROJECT_NAME} PRIVATE
    STUB_DRIVER_DIR="${STUB_DRIVER_DIR}"
    STUB_DRIVER_NAME="${STUB_DRIVER_NAME}")
else()
  message("-- ERROR!: ${PROJECT_NAME} excluded! unmet dependencies")
  if(NOT HIP_FOUND)
    message(FATAL_ERROR "-- ERROR!: HIP Not Found!")
  endif()
  if(NOT rocdecode_FOUND)
    message(FATAL_ERROR "-- ERROR!: rocdecode Not Found!")
  endif()
  if(NOT EXISTS ${STUB_DRIVER_PREBUILT})
    message(FATAL_ERROR "-- ERROR!: prebuilt stub VA driver ${STUB_DRIVER_PREBUILT} Not Found!")
  endif()
endif()
