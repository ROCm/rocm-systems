###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# Builds HSACOs for the device_bitcode functional test (kernel + rocshmem
# device bitcode). Included from tests/functional_tests/CMakeLists.txt.
#
# Builds one HSACO per architecture in BITCODE_GPU_ARCHS (set by DeviceBitcode.cmake).
# At runtime, device_bitcode_tester.cpp selects the HSACO matching the local GPU
# and skips gracefully if none is found.

# Verify rocshmem_device_bitcode target exists (created by DeviceBitcode.cmake)
if(NOT TARGET rocshmem_device_bitcode)
  message(WARNING "device_bitcode_tester: rocshmem_device_bitcode target not found. "
    "HSACOs will not be built; test will skip at runtime.")
  return()
endif()

if(NOT BITCODE_GPU_ARCHS)
  message(WARNING "device_bitcode_tester: BITCODE_GPU_ARCHS is empty. "
    "HSACOs will not be built; test will skip at runtime.")
  return()
endif()

# LLVM_CLANG and LLVM_LINK are already set by DeviceBitcode.cmake.
# Search for additional tools needed for HSACO generation.
find_program(LLVM_LLC llc PATHS ${ROCM_PATH}/llvm/bin NO_DEFAULT_PATH QUIET)
find_program(LLVM_LLD ld.lld PATHS ${ROCM_PATH}/llvm/bin NO_DEFAULT_PATH QUIET)

if(NOT LLVM_LLC OR NOT LLVM_LLD)
  message(WARNING "device_bitcode_tester: llc/ld.lld not found (ROCM_PATH=${ROCM_PATH}). "
    "HSACOs will not be built; test will skip at runtime.")
  return()
endif()

# --- HSACO build steps --------------------------------------------------------

set(ALL_TESTER_HSACOS "")

foreach(GPU_ARCH ${BITCODE_GPU_ARCHS})
  set(KERNEL_BC  ${CMAKE_CURRENT_BINARY_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}.bc)
  set(LINKED_BC  ${CMAKE_CURRENT_BINARY_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}_linked.bc)
  set(OBJ_FILE   ${CMAKE_CURRENT_BINARY_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}.o)
  set(HSACO_FILE ${CMAKE_CURRENT_BINARY_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}.hsaco)
  set(DEVICE_LIB ${CMAKE_BINARY_DIR}/librocshmem_device_${GPU_ARCH}.bc)

  add_custom_command(
    OUTPUT ${KERNEL_BC}
    COMMAND ${LLVM_CLANG}
      -x hip --cuda-device-only -std=c++20 -emit-llvm
      --offload-arch=${GPU_ARCH}
      -fvisibility=default
      -c ${CMAKE_CURRENT_SOURCE_DIR}/device_bitcode_tester_kernel.hip
      -o ${KERNEL_BC}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/device_bitcode_tester_kernel.hip
    COMMENT "device_bitcode_tester: compiling kernel for ${GPU_ARCH}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${LINKED_BC}
    COMMAND ${LLVM_LINK}
      ${KERNEL_BC}
      --override=${DEVICE_LIB}
      -o ${LINKED_BC}
    DEPENDS ${KERNEL_BC} ${DEVICE_LIB} rocshmem_device_bitcode
    COMMENT "device_bitcode_tester: linking with device bitcode for ${GPU_ARCH}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${LLVM_LLC}
      -mtriple=amdgcn-amd-amdhsa
      -mcpu=${GPU_ARCH}
      --amdgpu-internalize-symbols=false
      -filetype=obj
      ${LINKED_BC}
      -o ${OBJ_FILE}
    DEPENDS ${LINKED_BC}
    COMMENT "device_bitcode_tester: compiling to object for ${GPU_ARCH}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${HSACO_FILE}
    COMMAND ${LLVM_LLD} -shared ${OBJ_FILE} -o ${HSACO_FILE}
    DEPENDS ${OBJ_FILE}
    COMMENT "device_bitcode_tester: linking HSACO for ${GPU_ARCH}"
    VERBATIM
  )

  list(APPEND ALL_TESTER_HSACOS ${HSACO_FILE})

  rocm_install(FILES ${HSACO_FILE} COMPONENT tests
    DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)

endforeach()

add_custom_target(device_bitcode_tester_hsacos ALL
  DEPENDS ${ALL_TESTER_HSACOS}
)

add_dependencies(${PROJECT_NAME} device_bitcode_tester_hsacos)

message(STATUS "Device bitcode test (in rocshmem_functional_tests) enabled for: ${BITCODE_GPU_ARCHS}")
