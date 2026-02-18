###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

###############################################################################
# ROCSHMEM DEVICE LIBRARY BUILD
###############################################################################
# This module builds rocSHMEM device libraries (bitcode and/or HSACO)
# for use with Triton-distributed JIT compilation and PyTorch integration.
#
# The device library enables:
# 1. JIT linking with Triton kernels (bitcode .bc files)
# 2. Module loading for PyTorch symmetric memory (HSACO files)
#
# Output files (per architecture):
#   - librocshmem_device_<arch>.bc    (LLVM bitcode)
#   - librocshmem_device_<arch>.hsaco (AMD GPU code object) [optional]
###############################################################################

option(BUILD_DEVICE_LIBRARY "Build device bitcode/HSACO library" ON)
option(BUILD_DEVICE_BITCODE "Build device library as LLVM bitcode (.bc)" ON)
option(BUILD_DEVICE_HSACO "Build device library as HSACO (.hsaco)" OFF)

if(NOT BUILD_DEVICE_LIBRARY)
  message(STATUS "Device library build disabled")
  return()
endif()

if(NOT BUILD_DEVICE_BITCODE AND NOT BUILD_DEVICE_HSACO)
  message(STATUS "Neither bitcode nor HSACO build enabled, skipping device library")
  return()
endif()

message(STATUS "Building rocSHMEM device library:")
message(STATUS "  - Bitcode (.bc): ${BUILD_DEVICE_BITCODE}")
message(STATUS "  - HSACO (.hsaco): ${BUILD_DEVICE_HSACO}")

###############################################################################
# DEVICE LIBRARY GPU TARGETS
###############################################################################
# Device library architectures - use simplified list (no xnack variants)
set(DEVICE_LIB_GPU_TARGETS "gfx90a" "gfx942" "gfx950" "gfx1100"
    CACHE STRING "GPU architectures for device library")

# Filter by ROCm version
if(${ROCM_MAJOR_VERSION} LESS 7)
  list(REMOVE_ITEM DEVICE_LIB_GPU_TARGETS "gfx950")
  message(STATUS "ROCm ${ROCM_MAJOR_VERSION}.x detected - excluding gfx950 from device library")
endif()

message(STATUS "Device library GPU targets: ${DEVICE_LIB_GPU_TARGETS}")

###############################################################################
# DEVICE SOURCE FILES
###############################################################################
# Core device-side sources that need to be included in the device library
# NOTE: Match Triton-distributed build exactly
set(DEVICE_SOURCES
  ${CMAKE_SOURCE_DIR}/src/rocshmem_gpu.cpp
  ${CMAKE_SOURCE_DIR}/src/context_device.cpp
  ${CMAKE_SOURCE_DIR}/src/team.cpp
  ${CMAKE_SOURCE_DIR}/src/util.cpp
  ${CMAKE_SOURCE_DIR}/src/ipc_policy.cpp
  # Files NOT in Triton build (excluded):
  # - atomic_return.cpp
  # - backend_bc.cpp (host-side only)
  # - wf_coal_policy.cpp
  # - memory/dlmalloc.cpp
)

# Add backend-specific device sources if enabled
if(USE_RO)
  list(APPEND DEVICE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/reverse_offload/backend_ro.cpp
    ${CMAKE_SOURCE_DIR}/src/reverse_offload/context_ro_device.cpp
  )
endif()

if(USE_IPC)
  list(APPEND DEVICE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/ipc/backend_ipc.cpp
    ${CMAKE_SOURCE_DIR}/src/ipc/context_ipc_device.cpp
    ${CMAKE_SOURCE_DIR}/src/ipc/context_ipc_device_coll.cpp
  )
endif()

if(USE_GDA)
  list(APPEND DEVICE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/gda/backend_gda.cpp
    ${CMAKE_SOURCE_DIR}/src/gda/context_gda_device.cpp
    ${CMAKE_SOURCE_DIR}/src/gda/queue_pair.cpp
    # TODO: Fix ionic/mlx5 compilation errors
    # ${CMAKE_SOURCE_DIR}/src/gda/ionic/queue_pair_ionic.cpp
    # ${CMAKE_SOURCE_DIR}/src/gda/mlx5/queue_pair_mlx5.cpp
    # ${CMAKE_SOURCE_DIR}/src/gda/mlx5/segment_builder.cpp
  )
endif()

# Add sync sources
list(APPEND DEVICE_SOURCES
  ${CMAKE_SOURCE_DIR}/src/sync/abql_block_mutex.cpp
)

# Add device wrapper (Triton entry point)
list(APPEND DEVICE_SOURCES
  ${CMAKE_SOURCE_DIR}/src/device_wrapper/rocshmem_device_wrapper.cpp
)

list(LENGTH DEVICE_SOURCES NUM_DEVICE_SOURCES)
message(STATUS "Device library includes ${NUM_DEVICE_SOURCES} source files")

###############################################################################
# COMPILER PATHS
###############################################################################
set(CLANGXX "${ROCM_PATH}/lib/llvm/bin/clang++")
set(LLVM_LINK "${ROCM_PATH}/lib/llvm/bin/llvm-link")
set(OPT "${ROCM_PATH}/lib/llvm/bin/opt")
set(LLC "${ROCM_PATH}/lib/llvm/bin/llc")
set(LD_LLD "${ROCM_PATH}/lib/llvm/bin/ld.lld")

# Verify tools exist
foreach(TOOL ${CLANGXX} ${LLVM_LINK} ${OPT} ${LLC} ${LD_LLD})
  if(NOT EXISTS "${TOOL}")
    message(FATAL_ERROR "Required LLVM tool not found: ${TOOL}")
  endif()
endforeach()

###############################################################################
# BUILD DEVICE LIBRARY FOR EACH ARCHITECTURE
###############################################################################
set(DEVICE_BITCODE_FILES)
set(DEVICE_HSACO_FILES)

foreach(GPU_TARGET ${DEVICE_LIB_GPU_TARGETS})
  set(BC_FILES_FOR_TARGET)
  set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/device_lib/${GPU_TARGET}")
  file(MAKE_DIRECTORY "${OUTPUT_DIR}")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

  # Build compiler flags
  set(DEVICE_COMPILE_FLAGS
    -x hip --cuda-device-only -std=c++20 -emit-llvm
    --offload-arch=${GPU_TARGET}
    -mcode-object-version=5
    -fgpu-rdc
    -I${CMAKE_SOURCE_DIR}/include
    -I${CMAKE_BINARY_DIR}/include
    -I${CMAKE_BINARY_DIR}/include/rocshmem
    -I${CMAKE_SOURCE_DIR}/src
  )

  if(USE_IPC)
    list(APPEND DEVICE_COMPILE_FLAGS -DUSE_IPC=1)
    # NOTE: ENABLE_IPC_BITCODE not set - causes issues with context types
  endif()
  if(USE_RO)
    list(APPEND DEVICE_COMPILE_FLAGS -DUSE_RO=1)
  endif()
  if(USE_GDA)
    list(APPEND DEVICE_COMPILE_FLAGS -DUSE_GDA=1)
    # NOTE: ENABLE_IBGDA_BITCODE not set - causes issues with context types
  endif()

  # Step 1: Compile each source file to bitcode
  foreach(SRC_FILE ${DEVICE_SOURCES})
    get_filename_component(SRC_NAME "${SRC_FILE}" NAME_WE)
    set(BC_FILE "${OUTPUT_DIR}/${SRC_NAME}.bc")

    add_custom_command(
      OUTPUT "${BC_FILE}"
      COMMAND ${CLANGXX}
        ${DEVICE_COMPILE_FLAGS}
        -c "${SRC_FILE}" -o "${BC_FILE}"
      DEPENDS "${SRC_FILE}" "${CMAKE_BINARY_DIR}/include/rocshmem/rocshmem_config.h"
      COMMENT "Compiling ${SRC_NAME}.cpp to bitcode for ${GPU_TARGET}"
      VERBATIM
    )

    list(APPEND BC_FILES_FOR_TARGET "${BC_FILE}")
  endforeach()

  # Step 2: Link all bitcode files for this architecture
  set(LINKED_BC "${OUTPUT_DIR}/librocshmem_device_${GPU_TARGET}.linked.bc")
  add_custom_command(
    OUTPUT "${LINKED_BC}"
    COMMAND ${LLVM_LINK}
      ${BC_FILES_FOR_TARGET}
      -o "${LINKED_BC}"
    DEPENDS ${BC_FILES_FOR_TARGET}
    COMMENT "Linking bitcode for ${GPU_TARGET}"
    VERBATIM
  )

  # Step 3: Copy linked bitcode to final location (no optimization to preserve symbols)
  # NOTE: Optimization removed to match Triton-distributed build and preserve symbols like print_lock
  set(FINAL_BC "${CMAKE_BINARY_DIR}/lib/librocshmem_device_${GPU_TARGET}.bc")
  add_custom_command(
    OUTPUT "${FINAL_BC}"
    COMMAND ${CMAKE_COMMAND} -E copy
      "${LINKED_BC}"
      "${FINAL_BC}"
    DEPENDS "${LINKED_BC}"
    COMMENT "Copying linked bitcode for ${GPU_TARGET}"
    VERBATIM
  )

  if(BUILD_DEVICE_BITCODE)
    list(APPEND DEVICE_BITCODE_FILES "${FINAL_BC}")
  endif()

  # Step 4 (optional): Generate HSACO from bitcode
  if(BUILD_DEVICE_HSACO)
    set(OBJ_FILE "${OUTPUT_DIR}/librocshmem_device_${GPU_TARGET}.o")
    set(HSACO_FILE "${CMAKE_BINARY_DIR}/lib/librocshmem_device_${GPU_TARGET}.hsaco")

    # Compile bitcode to object file
    add_custom_command(
      OUTPUT "${OBJ_FILE}"
      COMMAND ${LLC}
        -mtriple=amdgcn-amd-amdhsa
        -mcpu=${GPU_TARGET}
        -filetype=obj
        "${FINAL_BC}"
        -o "${OBJ_FILE}"
      DEPENDS "${FINAL_BC}"
      COMMENT "Compiling bitcode to object for ${GPU_TARGET}"
      VERBATIM
    )

    # Link object to HSACO
    add_custom_command(
      OUTPUT "${HSACO_FILE}"
      COMMAND ${LD_LLD} -shared
        "${OBJ_FILE}"
        -o "${HSACO_FILE}"
      DEPENDS "${OBJ_FILE}"
      COMMENT "Linking HSACO for ${GPU_TARGET}"
      VERBATIM
    )

    list(APPEND DEVICE_HSACO_FILES "${HSACO_FILE}")
  endif()
endforeach()

###############################################################################
# CREATE DEVICE LIBRARY TARGET
###############################################################################
add_custom_target(rocshmem_device_library
  DEPENDS ${DEVICE_BITCODE_FILES} ${DEVICE_HSACO_FILES}
  COMMENT "Building rocSHMEM device library"
)

# Make device library part of the main build
if(NOT BUILD_TESTS_ONLY)
  add_dependencies(${PROJECT_NAME} rocshmem_device_library)
endif()

###############################################################################
# INSTALLATION
###############################################################################
if(DEVICE_BITCODE_FILES)
  install(
    FILES ${DEVICE_BITCODE_FILES}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT device_library
  )
  message(STATUS "Device bitcode will be installed to: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()

if(DEVICE_HSACO_FILES)
  install(
    FILES ${DEVICE_HSACO_FILES}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT device_library
  )
  message(STATUS "Device HSACO will be installed to: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()

message(STATUS "Device library build configured successfully")
