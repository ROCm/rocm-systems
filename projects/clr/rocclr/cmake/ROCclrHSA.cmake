# Copyright (c) 2020 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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

if(UNIX)
  find_package(hsa-runtime64 1.11 REQUIRED CONFIG
    PATHS
      /opt/rocm/
      ${ROCM_INSTALL_PATH}
    PATH_SUFFIXES
      cmake/hsa-runtime64
      lib/cmake/hsa-runtime64
      lib64/cmake/hsa-runtime64)
else()
  find_package(hsa-runtime64 CONFIG
    PATHS
      /opt/rocm/
      ${ROCM_INSTALL_PATH}
      ${CMAKE_CURRENT_BINARY_DIR}
      ${CMAKE_INSTALL_PREFIX}
      ${CMAKE_INSTALL_PREFIX}/..
    PATH_SUFFIXES
      rocr/lib/cmake/hsa-runtime64
      rocr/runtime/hsa-runtime
      cmake/hsa-runtime64
      lib/cmake/hsa-runtime64
      lib64/cmake/hsa-runtime64)
endif()

if (ROCR_DLL_LOAD)
  find_path(AMD_HSA_INCLUDE_DIR hsa.h
    HINTS
      /opt/rocm
      ${ROCM_INSTALL_PATH}
      ${CMAKE_CURRENT_BINARY_DIR}
    PATHS
      ${CMAKE_CURRENT_BINARY_DIR}/..
      ${CMAKE_CURRENT_BINARY_DIR}/../..
      ${CMAKE_CURRENT_BINARY_DIR}/../../rocr
      ${ROCCLR_SRC_DIR}/../../rocr-runtime/runtime/hsa-runtime
    PATH_SUFFIXES
      include
      include/hsa
      inc)
  message("Roc CLR: " ${ROCCLR_SRC_DIR} "; HSA headers:" ${AMD_HSA_INCLUDE_DIR})
  target_compile_definitions(rocclr PUBLIC ROCR_DYN_DLL)
  target_include_directories(rocclr PUBLIC ${AMD_HSA_INCLUDE_DIR})
else()
  target_link_libraries(rocclr PUBLIC hsa-runtime64::hsa-runtime64)
endif()

#target_include_directories(rocclr PRIVATE ${AMD_HSA_INCLUDE_DIR}/..)

find_package(NUMA)
if(NUMA_FOUND)
  target_compile_definitions(rocclr PUBLIC ROCCLR_SUPPORT_NUMA_POLICY)
  target_include_directories(rocclr PUBLIC ${NUMA_INCLUDE_DIR})
  target_link_libraries(rocclr PUBLIC ${NUMA_LIBRARIES})
endif()

find_package(OpenGL REQUIRED)

target_sources(rocclr PRIVATE
  ${ROCCLR_SRC_DIR}/device/rocm/rocappprofile.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocrctx.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocblit.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocblitcl.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/roccounters.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocdevice.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocglinterop.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rockernel.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocmemory.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocprintf.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocprogram.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocsettings.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocsignal.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocvirtual.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocurilocator.cpp)

target_compile_definitions(rocclr PUBLIC WITH_HSA_DEVICE)
